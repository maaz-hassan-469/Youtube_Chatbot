from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace,HuggingFaceEmbeddings,HuggingFaceEndpoint
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnablePassthrough
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
load_dotenv()

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

embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

prompt_template=PromptTemplate(
    template="""Answer the question based ONLY on the provided video transcript context.
    If you don't know the answer, say that you don't know based on this video.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:""",
    input_variables=['context','question']
)
parser=StrOutputParser()
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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

        print("Vectorstore ready for questions!")
     
        setup_and_retrieval = RunnableParallel(
            {
                "context": advanced_retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }
        )

        rag_chain = setup_and_retrieval | prompt_template | model | parser

        while(True):
            user_query = input(f"\n[Video: {video_id}] Ask a question (or type 'back' for new video): ").strip()
            if user_query.lower() in ["back","exit"]:
                break

            print("\nAnswer: ", end="", flush=True)
            for chunk in rag_chain.stream(user_query):
                 print(chunk, end="", flush=True)
            print()

    except Exception as e:
        print(f"\nError Details: {e}")
