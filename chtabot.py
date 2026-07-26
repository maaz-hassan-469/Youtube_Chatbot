from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace,HuggingFaceEmbeddings,HuggingFaceEndpointEmbeddings
# from langchain_huggingface import ChatHuggingFace,HuggingFaceEmbeddings,HuggingFaceEndpointEmbeddings
# from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder 
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnablePassthrough
from langchain_core.messages import HumanMessage,AIMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
# while(True):
#     video_id=input("enter a video_id to ask question:")
#     if video_id.lower()=="exit":
#         break
#     try:
#         transcript_list=YouTubeTranscriptApi.get_transcript(video_id,languages=["en"])
#         transcript=" ".join(chunk["text"] for chunk in transcript_list)
#         print("\n--- Transcript Retrieved Successfully ---")
#         print(transcript)
#     except:
#         print("no captions available for this video")
# llm=HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Flash",
#                         task="text-generation",
#                         temperature=0.3)
# model1=ChatHuggingFace(llm=llm)

model= ChatGroq(
    model_name="llama-3.1-8b-instant", 
    temperature=0.2
)

# embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding=HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

prompt_template=ChatPromptTemplate.from_messages([
("system","""Answer the question based ONLY on the provided video transcript context.
If you don't know the answer, say that you don't know based on this video.
Context:
{context}
"""),
MessagesPlaceholder(variable_name="chat_history"),
("human","{question}")
])
parser=StrOutputParser()
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chat_memories={}

while(True):
    video_id=input("\n==================================================\nEnter Video ID to process (or type 'exit' to quit): ")
    if video_id.lower()=="exit":
        break
    formatted_url = f"https://www.youtube.com/watch?v={video_id}"
    print(f"🔎 Fetching transcript for Video ID: '{video_id}'...")
    try:
        vector_store=Chroma(
            persist_directory="./chroma_db",
            embedding_function=embedding,
            collection_name=f"yt_{video_id.lower()}"
        )

        existing_docs_count=len(vector_store.get()["ids"])
        if existing_docs_count>0:
            print(f"Found existing embeddings on disk for Video ID '{video_id}' ({existing_docs_count} vectors loaded instantly)!")
        else:
            loader=YoutubeLoader.from_youtube_url(
                formatted_url,
                add_video_info=False,
                language=["en","en-US","en-GB","hi"]
                )

            docs=loader.load()
            print("Transcript fetched successfully!")

            splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=200)

            chunks=splitter.split_documents(docs)
            print(f"Total Text Chunks Created: {len(chunks)}")

            vector_store=Chroma.from_documents(
                persist_directory="./chroma_db",
                documents=chunks,
                embedding=embedding,
                collection_name=f"yt_{video_id.lower()}"
             )


        base_retriever = vector_store.as_retriever(search_type="mmr",search_kwargs={"k": 3,"fetch_k": 10, "lambda_mult": 0.5})
        advanced_retriever=MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=model
        )

        if video_id not in chat_memories:
            chat_memories[video_id]=[]

        print("Vectorstore ready for questions!")

        while(True):
            user_query = input(f"\n[Video: {video_id}] Ask a question (or type 'back' for new video): ").strip()
            if user_query.lower() in ["back","exit"]:
                break

            retrieved_docs = advanced_retriever.invoke(user_query)
            context_text = format_docs(retrieved_docs)

            current_history=chat_memories[video_id]

            formatted_prompt = prompt_template.format_messages(
                context=context_text,
                chat_history=current_history,
                question=user_query
            )

            print("\nAnswer: ", end="", flush=True)
            full_response = ""
            for chunk in model.stream(formatted_prompt):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                full_response += content
                print(content, end="", flush=True)
            print()

            # 5. Append User Query & Model Answer to Video's History List in Dictionary
            chat_memories[video_id].append(HumanMessage(content=user_query))
            chat_memories[video_id].append(AIMessage(content=full_response))

    except Exception as e:
        print(f"\nError Details: {e}")
