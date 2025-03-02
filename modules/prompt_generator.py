import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables (in case this module is used directly)
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=api_key)

def generate_paragraph(topic, context="general"):
    """
    Generate a paragraph about the given topic using Gemini API, tailored to specific speaking contexts.
    
    Args:
        topic (str): The topic to generate content about
        context (str): The speaking context (interview, debate, etc.)
        
    Returns:
        str: A paragraph (3-5 sentences) about the topic optimized for the specified context
    """
    try:
        # First, list available models to see what's accessible
        print(f"Generating content for topic: '{topic}' in '{context}' speaking context")
        models = genai.list_models()
        available_models = []
        
        for model in models:
            print(f" - {model.name}")
            available_models.append(model.name)
        
        # Select the appropriate model - try a few options in order of preference
        model_options = [
            'models/gemini-1.5-pro',
            'models/gemini-1.0-pro',
            'gemini-pro',
            'gemini-1.0-pro',
            'gemini-1.5-pro'
        ]
        
        selected_model = None
        for model_name in model_options:
            if model_name in available_models or any(model_name in m for m in available_models):
                selected_model = model_name
                break
                
        if not selected_model:
            # Use the first available model that contains "gemini" and can generate content
            for model_name in available_models:
                if "gemini" in model_name.lower():
                    selected_model = model_name
                    break
        
        if not selected_model:
            # If still no model, try the first text model available
            if len(available_models) > 0:
                selected_model = available_models[0]
            else:
                raise ValueError("No available Gemini models found")
        
        print(f"Using model: {selected_model}")
        model = genai.GenerativeModel(selected_model)
        
        # Create context-specific prompts
        if context == "interview":
            prompt = f"""Generate a paragraph (4-6 sentences) about {topic} that would be perfect for practicing interview responses.

            The paragraph should:
            - Demonstrate professional expertise on {topic}
            - Include 1-2 specific accomplishments or experiences related to {topic}
            - Incorporate 1-2 industry-relevant terms or concepts
            - End with a forward-looking statement showing growth mindset
            - Be 100-130 words in length
            - Use clear, articulate language suitable for a job interview
            - Include content that allows the speaker to demonstrate confidence and competence
            
            Format as a single focused paragraph without headers or bullet points - just a clean, well-structured response that a job candidate would deliver.
            """
            
        elif context == "debate":
            prompt = f"""Generate a persuasive paragraph (4-6 sentences) presenting a well-reasoned position on {topic} suitable for debate practice.

            The paragraph should:
            - Present a clear stance on {topic} with 2-3 supporting arguments
            - Include at least one compelling statistic or evidence point
            - Acknowledge potential counterarguments in a strategic way
            - Use rhetorical techniques like parallel structure or rhetorical questions
            - Be 100-130 words in length
            - Use language that's assertive but not aggressive
            - Include content that allows for emphasis, strategic pauses, and vocal modulation
            
            Format as a single focused paragraph without headers or bullet points - just a clean, well-structured argument that would be effective in a debate setting.
            """
            
        elif context == "storytelling":
            prompt = f"""Generate an engaging narrative paragraph (4-6 sentences) about {topic} that would be perfect for storytelling practice.

            The paragraph should:
            - Begin with a hook that creates interest in {topic}
            - Include vivid descriptive language and sensory details
            - Incorporate emotional elements or a character perspective
            - Build toward a mini-climax or revelation
            - Be 100-130 words in length
            - Use varied sentence structures with opportunities for dramatic pauses
            - Include content that allows for expressive vocal variation and emotional delivery
            
            Format as a single evocative paragraph without headers or bullet points - just a clean, well-structured story snippet that would captivate listeners.
            """
            
        elif context == "business_presentation":
            prompt = f"""Generate a professional paragraph (4-6 sentences) about {topic} that would be perfect for business presentation practice.

            The paragraph should:
            - Begin with a clear statement of the business relevance of {topic}
            - Include 1-2 data points or market insights related to {topic}
            - Identify a specific business opportunity or challenge
            - End with an actionable conclusion or recommendation
            - Be 100-130 words in length
            - Use professional business vocabulary appropriate for executives
            - Include content with clear key points that allow for emphasis and strategic pauses
            
            Format as a single focused paragraph without headers or bullet points - just a clean, well-structured business insight that would be delivered in a corporate setting.
            """
            
        elif context == "casual":
            prompt = f"""Generate a conversational paragraph (4-6 sentences) about {topic} that would sound natural in a casual social setting.

            The paragraph should:
            - Have a friendly, relaxed tone when discussing {topic}
            - Include a personal observation or light anecdote
            - Use conversational language with occasional colloquialisms
            - Invite further discussion with an open-ended thought
            - Be 90-120 words in length
            - Flow naturally as everyday speech would
            - Include content that allows for natural expressions and conversational rhythm
            
            Format as a single authentic-sounding paragraph without headers or bullet points - just a clean, well-structured casual remark that would feel natural in a social conversation.
            """
            
        elif context == "teaching":
            prompt = f"""Generate an instructional paragraph (4-6 sentences) about {topic} that would be perfect for teaching or training practice.

            The paragraph should:
            - Begin with a clear introduction of the concept related to {topic}
            - Include 1-2 key principles or facts that learners should remember
            - Provide a simple example or application to illustrate the concept
            - End with a connection to broader understanding or practical use
            - Be 100-130 words in length
            - Use clear, accessible language with 1-2 field-specific terms explained simply
            - Include content structured for clear emphasis on key learning points
            
            Format as a single focused paragraph without headers or bullet points - just a clean, well-structured explanation that an educator would deliver.
            """
            
        else:  # general context
            prompt = f"""Generate a well-structured paragraph (4-6 sentences) about the topic: {topic}.

            The paragraph should:
            - Present a balanced overview of {topic} with 2-3 interesting aspects
            - Include at least one surprising or thought-provoking point
            - Balance informative content with engaging delivery
            - End with a satisfying conclusion or thought-provoking question
            - Be 100-130 words in length
            - Use clear, articulate language with good rhythm
            - Include content with natural opportunities for vocal emphasis and expression
            
            Format as a single engaging paragraph without headers or bullet points - just a clean, well-structured passage that would be satisfying to read aloud.
            """
        
        # Generate content
        response = model.generate_content(prompt)
        
        # Extract text from response
        if hasattr(response, 'text'):
            paragraph = response.text
        elif hasattr(response, 'parts') and len(response.parts) > 0:
            paragraph = response.parts[0].text
        else:
            # Try accessing through different attributes based on API version
            try:
                paragraph = str(response)
            except:
                raise ValueError("Could not extract text from API response")
        
        # Clean up the response if necessary
        paragraph = paragraph.strip()
        paragraph = paragraph.replace('"', '')
        
        # Simple validation
        if len(paragraph.split()) < 20:
            print(f"Warning: Generated paragraph is too short ({len(paragraph.split())} words)")
            raise ValueError("Generated paragraph is too short")
            
        print(f"Successfully generated paragraph of {len(paragraph.split())} words for {context} context")
        return paragraph
    
    except Exception as e:
        print(f"Error generating paragraph: {str(e)}")
        
        # Detailed error logging
        import traceback
        traceback.print_exc()
        
        # Fallback paragraphs based on context
        fallbacks = {
            "interview": f"In my career, I've developed substantial expertise in {topic} through both education and hands-on experience. I've successfully implemented {topic}-related strategies that resulted in measurable improvements for my team and organization. My approach to {topic} combines established best practices with innovative thinking to drive results. I'm particularly proud of how I've used {topic} to solve complex challenges, and I'm eager to bring these skills to new opportunities in this field.",
            
            "debate": f"The evidence overwhelmingly demonstrates that {topic} represents a critical issue requiring immediate attention. Studies from leading researchers show that {topic} impacts multiple aspects of society, with particularly significant implications for future generations. While some may argue that other priorities should take precedence, the data clearly indicates that addressing {topic} yields substantial benefits across numerous domains. Therefore, any responsible approach must prioritize {topic} as an essential component of a comprehensive solution strategy.",
            
            "storytelling": f"I'll never forget my first encounter with {topic} on that unusually warm autumn afternoon. The sunlight filtered through the trees as the world of {topic} suddenly opened before me like a book whose pages contained both mystery and revelation. The fascinating details and unexpected connections within {topic} captured my imagination immediately, revealing patterns I had never noticed in everyday life. What began as casual curiosity evolved into a journey that would transform my understanding of not just {topic}, but the interconnected nature of all things.",
            
            "business_presentation": f"Our analysis reveals that {topic} represents a significant market opportunity in the coming fiscal year. Current industry trends show a 15% year-over-year growth in {topic}-related sectors, with particularly strong performance in emerging markets. Our company is uniquely positioned to leverage our existing expertise in {topic} to capture market share from less specialized competitors. I recommend allocating additional resources to our {topic} initiative, as projections indicate potential revenue increases of 22% with relatively modest additional investment.",
            
            "casual": f"You know, I was just thinking about {topic} the other day when I saw something online that completely surprised me. It's funny how {topic} shows up in so many unexpected places once you start paying attention to it. My friend Jamie is really into {topic} and always shares these interesting little facts that I'd never have known otherwise. Have you ever noticed how people's perspectives on {topic} can be so different depending on their experiences?",
            
            "teaching": f"Today, we'll explore the fundamental concepts of {topic}, which plays a crucial role in understanding our broader subject area. The two key principles to remember about {topic} are its structure and function, which work together to create the effects we observe. For example, when we examine {topic} in real-world scenarios, we see these principles applied to solve specific problems or create new opportunities. Understanding {topic} will provide you with tools that apply across multiple contexts in your future studies and professional work.",
            
            "general": f"The topic of {topic} is fascinating and worth exploring in detail. It has many interesting aspects that we can discuss and analyze. Understanding {topic} better can help us appreciate its significance in our daily lives. Many experts have studied {topic} and provided valuable insights that enrich our knowledge. As we continue to learn about {topic}, we discover new perspectives and applications."
        }
        
        context_fallback = fallbacks.get(context, fallbacks["general"])
        return context_fallback
        
# Example usage
if __name__ == "__main__":
    test_topic = "climate change"
    paragraph = generate_paragraph(test_topic)
    print("\nGenerated paragraph:")
    print(paragraph)