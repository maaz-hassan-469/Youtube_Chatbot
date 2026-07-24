from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace,HuggingFaceEmbeddings,HuggingFaceEndpoint
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader

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

    except Exception as e:
        print(f"\nError Details: {e}")




        




