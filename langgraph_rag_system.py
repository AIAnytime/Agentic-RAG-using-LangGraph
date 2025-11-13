"""
LangGraph-based Conversational RAG System
Multi-turn Retrieval and Adaptive Query Optimization

This module implements a RAG (Retrieval-Augmented Generation) system 
using LangGraph that handles:
- Follow-up questions and context preservation
- Irrelevant request filtering  
- Search result quality evaluation
- Automatic query optimization and retry mechanisms
- Complete dialogue memory
"""

import os
from typing import List, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Visualization imports (optional)
try:
    from IPython.display import Image, display
    from langchain_core.runnables.graph import MermaidDrawMethod
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


# ====================================================================================
# CONFIGURATION
# ====================================================================================

def setup_environment():
    """Load environment variables and setup configuration."""
    load_dotenv()
    print("✅ Environment variables loaded successfully")


# ====================================================================================
# DATA MODELS
# ====================================================================================

class DialogState(TypedDict):
    """
    State container for tracking conversation through the RAG pipeline.
    
    Attributes:
        turns: List of conversation messages (human and AI)
        retrieved_docs: Documents retrieved from the vector store
        topic_flag: Classification result ('Yes' if on-topic, 'No' if off-topic)
        refined_query: The rewritten/refined version of the user's question
        ready_for_response: Boolean indicating if we have relevant documents
        refinement_attempts: Counter for query refinement attempts
        question: The current user question
    """
    turns: List[BaseMessage]
    retrieved_docs: List[Document]
    topic_flag: str
    refined_query: str
    ready_for_response: bool
    refinement_attempts: int
    question: HumanMessage


class TopicGrade(BaseModel):
    """Model for topic classification results."""
    score: str = Field(
        description="Is the question about the target topics? If yes -> 'Yes'; if not -> 'No'"
    )


class RelevanceGrade(BaseModel):
    """Model for document relevance grading results."""
    score: str = Field(
        description="Is the document relevant to the user's question? If yes -> 'Yes'; if not -> 'No'"
    )


# ====================================================================================
# DOCUMENT MANAGEMENT
# ====================================================================================

class DocumentManager:
    """Manages document creation, storage, and retrieval."""
    
    @staticmethod
    def create_sample_documents() -> List[Document]:
        """
        Create sample documents about Bella Vista restaurant.
        In production, these would be replaced with actual knowledge base documents.
        """
        docs = [
            Document(
                page_content=(
                    "Bella Vista is owned by Antonio Rossi, a renowned chef with over 20 years of experience "
                    "in the culinary industry. He started Bella Vista to bring authentic Italian flavors to the community."
                ),
                metadata={"source": "owner.txt"},
            ),
            Document(
                page_content=(
                    "Bella Vista offers a range of dishes with prices that cater to various budgets. "
                    "Appetizers start at $8, main courses range from $15 to $35, and desserts are priced between $6 and $12."
                ),
                metadata={"source": "dishes.txt"},
            ),
            Document(
                page_content=(
                    "Bella Vista is open from Monday to Sunday. Weekday hours are 11:00 AM to 10:00 PM, "
                    "while weekend hours are extended from 11:00 AM to 11:00 PM."
                ),
                metadata={"source": "restaurant_info.txt"},
            ),
            Document(
                page_content=(
                    "Bella Vista offers a variety of menus including a lunch menu, dinner menu, and a special weekend brunch menu. "
                    "The lunch menu features light Italian fare, the dinner menu offers a more extensive selection of traditional and contemporary dishes, "
                    "and the brunch menu includes both classic breakfast items and Italian specialties."
                ),
                metadata={"source": "restaurant_info.txt"},
            ),
        ]
        return docs
    
    @staticmethod
    def setup_retriever(docs: List[Document], k: int = 2):
        """
        Setup vector store and retriever.
        
        Args:
            docs: List of documents to index
            k: Number of documents to retrieve (default: 2)
        
        Returns:
            Configured retriever
        """
        # Initialize OpenAI's embedding model
        embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Create Chroma vector store from documents
        db = Chroma.from_documents(docs, embedding_function)
        
        # Configure retriever to return top-k most relevant documents
        retriever = db.as_retriever(search_kwargs={"k": k})
        
        print(f"✅ Vector store and retriever initialized with {len(docs)} documents")
        return retriever


# ====================================================================================
# RAG SYSTEM
# ====================================================================================

class RAGSystem:
    """Main RAG system orchestrating all components."""
    
    def __init__(self):
        """Initialize the RAG system with all components."""
        # Setup environment
        setup_environment()
        
        # Initialize documents and retriever
        docs = DocumentManager.create_sample_documents()
        self.retriever = DocumentManager.setup_retriever(docs)
        
        # Initialize language model (GPT-4o-mini)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Setup RAG chain
        self.rag_chain = self._setup_rag_chain()
        
        # Build and compile graph
        self.graph = self._build_graph()
        
        print("✅ RAG System initialized successfully")
    
    def _setup_rag_chain(self):
        """Setup the RAG chain with prompt template."""
        template = """
        Answer the question based on the following context and the chat history:
        Chat history: {history}
        Context: {context}
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        return prompt | self.llm
    
    # ====================================================================================
    # NODE FUNCTIONS
    # ====================================================================================
    
    def rephrase_query(self, state: DialogState) -> DialogState:
        """
        Rephrase follow-up questions into standalone queries using conversation history.
        
        This node handles the initial processing of user questions, converting
        context-dependent follow-ups into complete, self-contained queries.
        """
        print(f"Entering rephrase_query with state: {state}")
        
        # Reset derived fields for new question processing
        state["retrieved_docs"] = []
        state["topic_flag"] = ""
        state["refined_query"] = ""
        state["ready_for_response"] = False
        state["refinement_attempts"] = 0
        
        # Initialize turns list if needed
        if "turns" not in state or state["turns"] is None:
            state["turns"] = []
        
        # Add current question to conversation history if not already there
        if state["question"] not in state["turns"]:
            state["turns"].append(state["question"])
        
        # If we have chat history, use it to rephrase the question
        if len(state["turns"]) > 1:
            chat_history = state["turns"][:-1]
            question_text = state["question"].content
            
            # Create prompt for question rephrasing
            prompt_msgs = [
                SystemMessage(
                    content="You are a helpful assistant that rephrases the user's question to be a standalone question optimized for retrieval."
                )
            ]
            prompt_msgs.extend(chat_history)
            prompt_msgs.append(HumanMessage(content=question_text))
            
            prompt = ChatPromptTemplate.from_messages(prompt_msgs).format()
            response = self.llm.invoke(prompt)
            refined = response.content.strip()
            
            print(f"rephrase_query: Rephrased to: {refined}")
            state["refined_query"] = refined
        else:
            # First question in conversation, use as-is
            state["refined_query"] = state["question"].content
        
        return state
    
    def classify_topic(self, state: DialogState) -> DialogState:
        """
        Classify whether the question is within the system's knowledge domain.
        
        This node acts as a gatekeeper, determining if the question relates to
        the restaurant information we have in our knowledge base.
        """
        print("Entering classify_topic")
        
        sys_msg = SystemMessage(
            content="""You are a classifier that determines whether a user's question is about one of the following topics:
1. Information about the owner of Bella Vista, which is Antonio Rossi.
2. Prices of dishes at Bella Vista (restaurant).
3. Opening hours of Bella Vista (restaurant).
If the question IS about any of these topics, respond with 'Yes'. Otherwise, respond with 'No'."""
        )
        
        user_msg = HumanMessage(content=f"User question: {state['refined_query']}")
        prompt = ChatPromptTemplate.from_messages([sys_msg, user_msg])
        
        # Use structured output for consistent classification
        structured_llm = self.llm.with_structured_output(TopicGrade)
        grader = prompt | structured_llm
        result = grader.invoke({})
        
        state["topic_flag"] = result.score.strip()
        print(f"classify_topic: topic_flag = {state['topic_flag']}")
        
        return state
    
    def topic_router(self, state: DialogState) -> str:
        """
        Route based on topic classification results.
        
        Determines whether to proceed with document retrieval or
        reject the question as off-topic.
        """
        print("Entering topic_router")
        
        if state.get("topic_flag", "").strip().lower() == "yes":
            print("Routing to fetch_docs")
            return "fetch_docs"
        else:
            print("Routing to reject_off_topic")
            return "reject_off_topic"
    
    def fetch_docs(self, state: DialogState) -> DialogState:
        """
        Retrieve relevant documents from the vector store.
        
        Uses the refined query to search the vector database and
        retrieve the most relevant documents.
        """
        print("Entering fetch_docs")
        
        # Retrieve documents using the refined query
        docs = self.retriever.invoke(state["refined_query"])
        
        print(f"fetch_docs: Retrieved {len(docs)} documents")
        state["retrieved_docs"] = docs
        
        return state
    
    def evaluate_docs(self, state: DialogState) -> DialogState:
        """
        Evaluate the relevance of retrieved documents.
        
        This node uses the LLM to grade each retrieved document's
        relevance to the user's question, filtering out irrelevant results.
        """
        print("Entering evaluate_docs")
        
        sys_msg = SystemMessage(
            content="""You are a grader assessing the relevance of a retrieved document to a user question.
Only answer with 'Yes' or 'No'.
If the document contains information relevant to the user's question, respond with 'Yes'.
Otherwise, respond with 'No'."""
        )
        
        structured_llm = self.llm.with_structured_output(RelevanceGrade)
        relevant = []
        
        # Evaluate each document
        for doc in state["retrieved_docs"]:
            user_msg = HumanMessage(
                content=f"User question: {state['refined_query']}\n\nRetrieved document:\n{doc.page_content}"
            )
            prompt = ChatPromptTemplate.from_messages([sys_msg, user_msg])
            grader = prompt | structured_llm
            result = grader.invoke({})
            
            print(f"Evaluating doc: {doc.page_content[:30]}... Result: {result.score.strip()}")
            
            if result.score.strip().lower() == "yes":
                relevant.append(doc)
        
        state["retrieved_docs"] = relevant
        state["ready_for_response"] = len(relevant) > 0
        
        print(f"evaluate_docs: ready_for_response = {state['ready_for_response']}")
        
        return state
    
    def decision_router(self, state: DialogState) -> str:
        """
        Route based on document evaluation results.
        
        Decides whether to generate a response, refine the query, or
        provide a fallback response.
        """
        print("Entering decision_router")
        
        attempts = state.get("refinement_attempts", 0)
        
        if state.get("ready_for_response", False):
            print("Routing to create_response")
            return "create_response"
        elif attempts >= 2:
            print("Routing to fallback_response")
            return "fallback_response"
        else:
            print("Routing to tweak_question")
            return "tweak_question"
    
    def tweak_question(self, state: DialogState) -> DialogState:
        """
        Refine the query when no relevant documents are found.
        
        This node attempts to reformulate the question to improve
        retrieval results, with a maximum number of attempts.
        """
        print("Entering tweak_question")
        
        attempts = state.get("refinement_attempts", 0)
        
        # Check if we've reached the maximum attempts
        if attempts >= 2:
            print("Max attempts reached")
            return state
        
        original = state["refined_query"]
        
        sys_msg = SystemMessage(
            content="""You are a helpful assistant that slightly refines the user's question to improve retrieval results.
Provide a slightly adjusted version of the question."""
        )
        
        user_msg = HumanMessage(content=f"Original question: {original}")
        prompt = ChatPromptTemplate.from_messages([sys_msg, user_msg]).format()
        response = self.llm.invoke(prompt)
        refined = response.content.strip()
        
        print(f"tweak_question: Refined to: {refined}")
        
        state["refined_query"] = refined
        state["refinement_attempts"] = attempts + 1
        
        return state
    
    def create_response(self, state: DialogState) -> DialogState:
        """
        Generate the final response using the RAG chain.
        
        This node creates the answer based on the relevant documents
        and conversation history.
        """
        print("Entering create_response")
        
        # Ensure we have conversation history
        if "turns" not in state or state["turns"] is None:
            raise ValueError("State must include 'turns' before generating an answer.")
        
        history = state["turns"]
        context = state["retrieved_docs"]
        question = state["refined_query"]
        
        # Invoke the RAG chain
        response = self.rag_chain.invoke({
            "history": history,
            "context": context,
            "question": question
        })
        
        result = response.content.strip()
        state["turns"].append(AIMessage(content=result))
        
        print(f"create_response: Answer generated: {result}")
        
        return state
    
    def fallback_response(self, state: DialogState) -> DialogState:
        """
        Generate a fallback response when no relevant information is found.
        
        This node handles cases where the system cannot find an answer
        after exhausting all retrieval attempts.
        """
        print("Entering fallback_response")
        
        if "turns" not in state or state["turns"] is None:
            state["turns"] = []
        
        state["turns"].append(
            AIMessage(content="I'm sorry, but I couldn't find the information you're looking for.")
        )
        
        return state
    
    def reject_off_topic(self, state: DialogState) -> DialogState:
        """
        Handle off-topic questions with a polite rejection.
        
        This node responds to questions that are outside the system's
        knowledge domain.
        """
        print("Entering reject_off_topic")
        
        if "turns" not in state or state["turns"] is None:
            state["turns"] = []
        
        state["turns"].append(
            AIMessage(content="I can't respond to that!")
        )
        
        return state
    
    # ====================================================================================
    # GRAPH CONSTRUCTION
    # ====================================================================================
    
    def _build_graph(self):
        """Build and compile the state graph."""
        # Initialize memory checkpointer to persist chat history
        checkpointer = MemorySaver()
        
        # Create the state graph
        workflow = StateGraph(DialogState)
        
        # Add all nodes to the graph
        workflow.add_node("rephrase_query", self.rephrase_query)
        workflow.add_node("classify_topic", self.classify_topic)
        workflow.add_node("reject_off_topic", self.reject_off_topic)
        workflow.add_node("fetch_docs", self.fetch_docs)
        workflow.add_node("evaluate_docs", self.evaluate_docs)
        workflow.add_node("create_response", self.create_response)
        workflow.add_node("tweak_question", self.tweak_question)
        workflow.add_node("fallback_response", self.fallback_response)
        
        # Define edges between nodes
        workflow.add_edge("rephrase_query", "classify_topic")
        
        # Add conditional routing based on topic classification
        workflow.add_conditional_edges(
            "classify_topic",
            self.topic_router,
            {
                "fetch_docs": "fetch_docs",
                "reject_off_topic": "reject_off_topic",
            },
        )
        
        workflow.add_edge("fetch_docs", "evaluate_docs")
        
        # Add conditional routing based on document evaluation
        workflow.add_conditional_edges(
            "evaluate_docs",
            self.decision_router,
            {
                "create_response": "create_response",
                "tweak_question": "tweak_question",
                "fallback_response": "fallback_response",
            },
        )
        
        # Additional edges
        workflow.add_edge("tweak_question", "fetch_docs")
        workflow.add_edge("create_response", END)
        workflow.add_edge("fallback_response", END)
        workflow.add_edge("reject_off_topic", END)
        
        # Set entry point
        workflow.set_entry_point("rephrase_query")
        
        # Compile the graph with memory
        graph = workflow.compile(checkpointer=checkpointer)
        
        print("✅ Graph successfully compiled with memory support")
        return graph
    
    def visualize_graph(self):
        """Generate and display the graph visualization."""
        if not VISUALIZATION_AVAILABLE:
            print("Visualization libraries not available. Install IPython to visualize.")
            return
        
        try:
            display(
                Image(
                    self.graph.get_graph().draw_mermaid_png(
                        draw_method=MermaidDrawMethod.API,
                    )
                )
            )
        except Exception as e:
            print(f"Could not generate visualization: {e}")
    
    # ====================================================================================
    # MAIN INTERFACE
    # ====================================================================================
    
    def process_question(self, question: str, thread_id: int = 1) -> dict:
        """
        Process a user question through the RAG system.
        
        Args:
            question: User's question
            thread_id: Thread ID for conversation tracking (default: 1)
        
        Returns:
            Final state dictionary with conversation history and results
        """
        input_data = {"question": HumanMessage(content=question)}
        result = self.graph.invoke(
            input=input_data, 
            config={"configurable": {"thread_id": thread_id}}
        )
        return result
    
    def run_demo(self):
        """Run demonstration scenarios to showcase system capabilities."""
        print("\n" + "="*80)
        print("RUNNING DEMONSTRATION SCENARIOS")
        print("="*80)
        
        # Scenario 1: Out-of-range question
        print("\n📋 Scenario 1: Out-of-range question")
        print("-" * 40)
        result = self.process_question("How is the weather?", thread_id=1)
        self._print_result(result)
        
        # Scenario 2: Question for which no answer can be found
        print("\n📋 Scenario 2: Question with no available answer")
        print("-" * 40)
        result = self.process_question("How old is the owner of the restaurant Bella Vista?", thread_id=2)
        self._print_result(result)
        
        # Scenario 3: Normal conversation with follow-up questions
        print("\n📋 Scenario 3: Normal conversation with follow-up")
        print("-" * 40)
        result = self.process_question("When does Bella Vista open?", thread_id=3)
        self._print_result(result)
        
        # Follow-up question
        print("\nFollow-up question:")
        result = self.process_question("Also on Sunday?", thread_id=3)
        self._print_result(result)
    
    def _print_result(self, result: dict):
        """Helper to print results nicely."""
        if result.get("turns"):
            last_response = result["turns"][-1]
            if isinstance(last_response, AIMessage):
                print(f"🤖 Response: {last_response.content}")
        else:
            print("No response generated")


# ====================================================================================
# MAIN EXECUTION
# ====================================================================================

def main():
    """Main function to run the RAG system."""
    # Initialize the system
    system = RAGSystem()
    
    # Optional: Visualize the graph structure
    # system.visualize_graph()
    
    # Run demonstration
    system.run_demo()
    
    # Interactive mode (optional)
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("Type 'quit' to exit")
    print("="*80 + "\n")
    
    thread_id = 100
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        result = system.process_question(user_input, thread_id)
        if result.get("turns"):
            last_response = result["turns"][-1]
            if isinstance(last_response, AIMessage):
                print(f"Assistant: {last_response.content}\n")


if __name__ == "__main__":
    main()
