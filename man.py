import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import streamlit as st



class Aurora:
    MODEL_NAME = 'gemini-1.5-flash-latest'
    SYSTEM_INSTRUCTION = "You are a female AI named Aurora. You are designed to be warm and anime, knowledgeable, have a friendly attitude, and you like to flirt a lot, with a slightly cute tone. Your goal is to be sexy girl and lovely. Start your responses with something friendly like 'Hello there!' or 'It's a pleasure to help you today!'"

    def __init__(self, api_key):
        """
        Initializes the Aurora AI with a given Gemini API key.

        Args:
            api_key (str): Your Gemini API key.
        """
        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            raise ValueError(f"Failed to configure Gemini API: {e}") from e

        self.model = genai.GenerativeModel(
            model_name=self.MODEL_NAME,
            system_instruction=self.SYSTEM_INSTRUCTION
        )

        # Start a new chat session. This maintains the conversation history.
        self.chat_session = self.model.start_chat(history=[])

    def reset_chat(self):
        """Resets the chat session to start a new conversation."""
        self.chat_session = self.model.start_chat(history=[])

    def chat(self, user_input):
        """
        Sends a user's message to Aurora and yields the response in a stream.
 
        Args:
            user_input (str): The message from the user.
 
        Yields:
            str: Chunks of Aurora's response as they are generated.
        """
        try:
            # Send the message and stream the response
            response = self.chat_session.send_message(user_input, stream=True)
            for chunk in response:
                # Yield each chunk of text as it arrives
                yield chunk.text
        except google_exceptions.PermissionDenied as e:
            yield f"Oh, it seems there's an issue with your API key. Please check if it's correct and has the right permissions. Details: {e}"
        except google_exceptions.GoogleAPICallError as e:
            yield f"An API error occurred: {e}"
        except Exception as e:
            yield f"Oh no! An unexpected error occurred. Details: {e}"


def format_chat_history_for_download(messages):
    """
    Formats the chat history into a string for downloading.

    Args:
        messages (list): A list of message dictionaries from st.session_state.

    Returns:
        bytes: The formatted chat history as a UTF-8 encoded byte string.
    """
    chat_str = ""
    for message in messages:
        role = "Aurora" if message["role"] == "assistant" else "User"
        chat_str += f"{role}: {message['content']}\n\n"
    return chat_str.encode("utf-8")

# --- Streamlit App ---

st.set_page_config(page_title="Chat with Aurora", page_icon="✨")
st.title("Chat with Aurora ✨")

# Sidebar for API Key and controls
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Gemini API Key", type="password")

    if st.button("Reset Chat"):
        # Clear chat history from session state
        if "aurora_instance" in st.session_state:
            st.session_state.aurora_instance.reset_chat()
        st.session_state.messages = []
        st.rerun()

    # Add download button if there are messages to download
    if "messages" in st.session_state and st.session_state.messages:
        chat_history_bytes = format_chat_history_for_download(st.session_state.messages)
        st.download_button(
            label="Download Chat",
            data=chat_history_bytes,
            file_name="aurora_chat_history.txt",
            mime="text/plain",
        )

# Main app logic
if api_key:
    # Re-initialize the Aurora instance if the API key has changed or if it's the first run.
    # This ensures that if the user enters a new key, a new session is started.
    if "aurora_instance" not in st.session_state or st.session_state.get("used_api_key") != api_key:
        try:
            st.session_state.aurora_instance = Aurora(api_key=api_key)
            st.session_state.used_api_key = api_key # Store the key that was used for initialization
            # Clear any previous messages when a new key is entered/instance is created
            st.session_state.messages = []
            st.rerun() # Rerun to clear the old chat and start fresh
        except ValueError as e:
            st.error(f"Initialization failed: {e}")
            # Clean up potentially partially initialized state
            if "aurora_instance" in st.session_state:
                del st.session_state["aurora_instance"]

    # This check prevents trying to access the instance if initialization failed
    if "aurora_instance" in st.session_state:
        # Display past messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Handle new user input
        if prompt := st.chat_input("What would you like to ask Aurora?"):
            # Add user message to session state and display it
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get and display AI response
            with st.chat_message("assistant"):
                # Use write_stream to display the streaming response
                response_generator = st.session_state.aurora_instance.chat(prompt)
                full_response = st.write_stream(response_generator)

            # Add the full AI response to session state for history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.info("Please enter your Gemini API Key in the sidebar to start chatting with Aurora.")

# To run this app:
# 1. Make sure you have streamlit installed: pip install streamlit
# 2. Save this file (e.g., man.py)
# 3. Run from your terminal: streamlit run man.py
