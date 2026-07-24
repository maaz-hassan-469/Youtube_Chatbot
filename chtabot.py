from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace,HuggingFaceEmbeddings,HuggingFaceEndpoint
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

while(True):
    video_id=input("enter a video_id to ask question:")
    if video_id.lower()=="exit":
        break
    try:
        transcript_list=YouTubeTranscriptApi.get_transcript(video_id,languages=["en"])
        transcript=" ".join(chunk["text"] for chunk in transcript_list)
        print("\n--- Transcript Retrieved Successfully ---")
        print(transcript)
    except:
        print("no captions available for this video")


        




