from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace,HuggingFaceEmbeddings,HuggingFaceEndpoint
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnablePassthrough
from dotenv import load_dotenv

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
llm=HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro",
                        task="text-generation",
                        temperature=0.3)
model=ChatHuggingFace(llm=llm)

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

while(True):
    video_id=input("\n==================================================\nEnter Video ID to process (or type 'exit' to quit): ")
    if video_id.lower()=="exit":
        break
    formatted_url = f"https://www.youtube.com/watch?v={video_id}"
    print(f"🔎 Fetching transcript for Video ID: '{video_id}'...")
    try:
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
        print("\n")
        print(chunks[0])


        vector_store=Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name="youtube_chatbot"
        )

        retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k": 3})
        print("Vectorstore ready for questions!")

        while(True):
            user_query = input(f"\n[Video: {video_id}] Ask a question (or type 'back' for new video): ").strip()
            if user_query.lower() in ["back","exit"]:
                break

            relevant_docs=retriever.invoke(user_query)
            context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

            prompt=prompt_template.invoke({"context":context_text,"question":user_query})
            response=model.invoke(prompt)
            print("\nAnswer:")
            print(response.content)

    except Exception as e:
        print(f"\nError Details: {e}")




        




