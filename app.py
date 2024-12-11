import os
import re
from typing import List, Dict
import google.generativeai as genai
import PyPDF2
import streamlit as st
import json

class PersonalAssistantBot:
    def __init__(self, pdf_path: str, api_key: str):
        """
        Initialize the personal assistant bot with advanced context management
        
        :param pdf_path: Path to the PDF containing personal information
        :param api_key: Google Gemini API key
        """
        # Configure Gemini API
        genai.configure(api_key=api_key)
        
        # Extract text from PDF
        self.personal_info = self._extract_pdf_text(pdf_path)
        
        # Initialize Gemini model
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Context management
        self.context_memory = {
            "conversation_tone": "friendly",
            "previous_topics": [],
            "user_preferences": self._extract_preferences()
        }
        
        # Predefined conversation templates
        self.conversation_templates = {
            "greeting": [
                "Hi there! I'm Srikanth's personal AI assistant. How can I help you today?",
                "Hello! I'm ready to assist you with any questions you might have.",
                "Greetings! What would you like to know about Srikanth?"
            ],
            "follow_up": [
                "Is there anything else you'd like to know?",
                "Feel free to ask me more!",
                "I'm happy to provide more details if you're interested."
            ]
        }
    
    def _extract_pdf_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF file with improved parsing
        
        :param pdf_path: Path to PDF file
        :return: Extracted text from PDF
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                full_text = ""
                for page in pdf_reader.pages:
                    full_text += page.extract_text() + "\n"
                return self._clean_extracted_text(full_text)
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean and structure extracted PDF text
        
        :param text: Raw extracted text
        :return: Cleaned and structured text
        """
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Basic text structuring
        text = text.replace('. ', '.\n')
        return text
    
    def _extract_preferences(self) -> Dict:
        """
        Extract user preferences from personal information
        
        :return: Dictionary of user preferences
        """
        # Look for key preference indicators in personal info
        preferences = {
            "communication_style": "professional" if "professional" in self.personal_info.lower() else "casual",
            "interests": self._find_interests(),
            "languages": self._find_languages()
        }
        return preferences
    
    def _find_interests(self) -> List[str]:
        """
        Extract interests from personal information
        
        :return: List of interests
        """
        interest_keywords = [
            "hobby", "interest", "passionate about", "enjoy", "like to"
        ]
        interests = []
        
        for keyword in interest_keywords:
            if keyword in self.personal_info.lower():
                # Naive extraction - could be improved with NLP
                match = re.findall(rf"{keyword}\s*:?\s*(.+?)[\n\.]", self.personal_info, re.IGNORECASE)
                interests.extend(match)
        
        return list(set(interests))
    
    def _find_languages(self) -> List[str]:
        """
        Extract known languages from personal information
        
        :return: List of languages
        """
        common_languages = [
            "English", "Spanish", "French", "German", "Mandarin", 
            "Arabic", "Hindi", "Portuguese", "Russian", "Japanese"
        ]
        
        return [lang for lang in common_languages if lang.lower() in self.personal_info.lower()]
    
    def generate_response(self, user_query: str, chat_history: List[Dict] = None) -> str:
        """
        Generate intelligent response with context awareness
        
        :param user_query: User's question
        :param chat_history: Previous conversation context
        :return: AI-generated response
        """
        # Prepare comprehensive context
        context_prompt = self._build_context_prompt(user_query, chat_history)
        
        # Dynamic response generation
        try:
            response = self.model.generate_content(context_prompt)
            
            # Update conversation context
            self._update_context(user_query)
            
            return response.text
        except Exception as e:
            return f"I'm having trouble processing that. Could you rephrase? Error: {e}"
    
    def _build_context_prompt(self, user_query: str, chat_history: List[Dict] = None) -> str:
        """
        Build a comprehensive context-aware prompt
        
        :param user_query: Current user query
        :param chat_history: Previous conversation context
        :return: Structured prompt for Gemini
        """
        context_parts = [
            f"Personal Information Profile:\n{self.personal_info}",
            f"User Preferences: {json.dumps(self.context_memory['user_preferences'])}",
            f"Conversation Tone: {self.context_memory['conversation_tone']}",
            f"Previous Conversation Topics: {', '.join(self.context_memory['previous_topics'])}",
            f"Current Query: {user_query}"
        ]
        
        if chat_history:
            context_parts.append("Recent Conversation History:")
            for msg in chat_history[-3:]:  # Include last 3 messages for context
                context_parts.append(f"{msg['role']}: {msg['content']}")
        
        comprehensive_prompt = "\n\n".join(context_parts) + "\n\nResponse Guidelines:\n" + """
        1. Be conversational and natural
        2. Refer to the personal information context
        3. Maintain a consistent tone
        4. Provide concise and relevant answers
        5. If unsure, ask for clarification
        """
        
        return comprehensive_prompt
    
    def _update_context(self, user_query: str):
        """
        Update conversation context after each interaction
        
        :param user_query: User's query to extract context
        """
        # Update previous topics
        topics = self._extract_topics(user_query)
        self.context_memory['previous_topics'].extend(topics)
        
        # Limit memory size
        self.context_memory['previous_topics'] = self.context_memory['previous_topics'][-5:]

    def _extract_topics(self, text: str) -> List[str]:
        """
        Extract key topics from text
        
        :param text: Input text
        :return: List of extracted topics
        """
        # Simple topic extraction - could be enhanced with NLP
        keywords = re.findall(r'\b\w{4,}\b', text.lower())
        return list(set(keywords))[:3]  # Return top 3 unique keywords
    
    def get_greeting(self) -> str:
        """
        Generate a random, personalized greeting
        
        :return: Greeting message
        """
        return self._get_random_item(self.conversation_templates['greeting'])
    
    def get_follow_up(self) -> str:
        """
        Generate a random follow-up prompt
        
        :return: Follow-up message
        """
        return self._get_random_item(self.conversation_templates['follow_up'])
    
    def _get_random_item(self, items: List[str]) -> str:
        """
        Get a random item from a list
        
        :param items: List of items
        :return: Randomly selected item
        """
        import random
        return random.choice(items)

def create_streamlit_interface(assistant: PersonalAssistantBot):
    """
    Create an enhanced Streamlit chat interface
    """
    st.set_page_config(page_title="Srikanth's AI Assistant", page_icon=":robot_face:")
    
    
    # Session state for messages and context
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Initial greeting
        st.session_state.messages.append({
            "role": "assistant", 
            "content": assistant.get_greeting()
        })
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Ask me anything about Srikanth"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            response = assistant.generate_response(
                prompt, 
                st.session_state.messages
            )
            st.markdown(response)
            
            # Optional follow-up suggestion
            if len(st.session_state.messages) % 3 == 0:
                follow_up = assistant.get_follow_up()
                st.markdown(f"*{follow_up}*")
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response
        })

def main():
    # Replace with your actual Google Gemini API key
    API_KEY = 'AIzaSyBz-r-DCBjQn2bU1AROkd490rmAixcQgAE'
    
    # Replace with the path to your personal information PDF
    PDF_PATH = 'SrikanthResume.pdf'
    
    # Create personal assistant instance
    assistant = PersonalAssistantBot(PDF_PATH, API_KEY)
    
    # Launch Streamlit interface
    create_streamlit_interface(assistant)

if __name__ == "__main__":
    main()