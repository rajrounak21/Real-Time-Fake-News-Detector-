import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.newspaper4k import Newspaper4k
from phi.tools.duckduckgo import DuckDuckGo
import google.generativeai as genai
import os
import re
from key import GOOGLE_API_KEY

# Set API Key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# Streamlit UI Setup
st.set_page_config(page_title="Fake News Predictor", page_icon="📰", layout="wide")
st.title("📰 Real-Time Fake News Detector")


@st.cache_resource
def initialize_news_agent():
    return Agent(
        name="Real Time Fake News Detection",
        model=Gemini(id="gemini-1.5-flash"),
        tools=[Newspaper4k(), DuckDuckGo()],
        debug_mode=False,  # Disabled to avoid showing raw API calls
        markdown=True
    )


multiagent_news_predict = initialize_news_agent()


# Function to check if input is a URL
def is_url(text):
    return re.match(r"https?://\S+", text) is not None


# User input (text or URL)
news_input = st.text_input("📌 Enter a news article or URL:")

# Verify News Button
if st.button("🔍 Verify News", key="news_verification_button"):
    if news_input:
        with st.spinner("⏳ Verifying news credibility..."):
            try:
                # Extract content if input is a URL
                if is_url(news_input):
                    try:
                        extracted_news = multiagent_news_predict.tools[0].extract_text(url=news_input)
                        news_text = extracted_news if extracted_news else None
                    except Exception as e:
                        st.error(f"❌ Error extracting content: {e}")
                        news_text = None
                else:
                    news_text = news_input

                # Stop execution if no valid content
                if not news_text:
                    st.warning("⚠️ No valid content to analyze. Please enter a different news article or URL.")
                    st.stop()

                # Generate AI verification response
                verification_prompt = f"""
                You are a fact-checking AI. Verify whether this news is real or fake:
                "{news_text}"

                🔍 **Steps to Verify:**
                - Search the latest reports using DuckDuckGo.
                - Compare the claim with trusted news sources.
                - Identify any inconsistencies, exaggerations, or false claims.

                **Response Format (Strictly Follow This):**
                - Verdict: "TRUE" or "FAKE"
                - Reason: [Provide a concise reason based on fact-checking]
                """
                response = multiagent_news_predict.run(verification_prompt)

                # Extract and display only the final verdict and reason
                response_text = response.content.strip()

                # Extract verdict and reason
                verdict_match = re.search(r"Verdict:\s*(TRUE|FAKE)", response_text, re.IGNORECASE)
                reason_match = re.search(r"Reason:\s*(.*)", response_text, re.IGNORECASE)

                if verdict_match:
                    verdict = verdict_match.group(1).upper()
                    if "TRUE" in verdict:
                        st.success(f"✅ **{verdict}**")
                    else:
                        st.error(f"🚨 **{verdict}**")

                    # Show reason if available
                    if reason_match:
                        reason = reason_match.group(1).strip()
                        st.info(f"🧐 **Reason:** {reason}")
                    else:
                        st.warning("⚠️ No reason provided by the AI.")
                else:
                    st.warning("⚠️ Unable to determine credibility.")

            except Exception as error:
                st.error(f"❌ An error occurred: {error}")
    else:
        st.warning("⚠️ Please enter a news article or URL for verification.")
