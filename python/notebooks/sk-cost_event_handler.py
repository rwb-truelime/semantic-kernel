# Install the required packages
# python -m pip install --upgrade semantic-kernel
# python -m pip install --upgrade pip
# python -m pip install --upgrade tiktoken
from enum import Enum
import asyncio
import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.prompt_template.input_variable import InputVariable
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments
import tiktoken
from semantic_kernel.events.function_invoking_event_args import FunctionInvokingEventArgs
from semantic_kernel.events.function_invoked_event_args import FunctionInvokedEventArgs


# Define a service enum to represent the available services
class Service(Enum):
    """
    Attributes:
        OpenAI (str): Represents the OpenAI service.
        AzureOpenAI (str): Represents the Azure OpenAI service.
        HuggingFace (str): Represents the HuggingFace service.
    """

    OpenAI = "openai"
    AzureOpenAI = "azureopenai"
    HuggingFace = "huggingface"


# Select a service to use for this notebook (available services: OpenAI, AzureOpenAI, HuggingFace)
selectedService = Service.AzureOpenAI

# Initiate Semantic Kernel
kernel = sk.Kernel()

# Add the selected service to the kernel
service_id = None
if selectedService == Service.OpenAI:
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

    api_key, org_id = sk.openai_settings_from_dot_env()
    service_id = "oai_chat_gpt"
    kernel.add_service(
        OpenAIChatCompletion(service_id=service_id, ai_model_id="gpt-3.5-turbo-1106", api_key=api_key, org_id=org_id),
    )
elif selectedService == Service.AzureOpenAI:
    from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

    deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
    service_id = "aoai_chat_completion"
    kernel.add_service(
        AzureChatCompletion(service_id=service_id, deployment_name=deployment, endpoint=endpoint, api_key=api_key),
    )


# Define a function to handle the function invoking event
def post_invocation_handler(kernel_function_metadata, event_args: FunctionInvokedEventArgs):
    '''
    This function is called after a function has been invoked.
    It extracts token counts from the function result and creates a nice table for you to enjoy 😍
    args:
        - kernel_function_metadata: The metadata of the function that was invoked.
        - event_args: The event arguments containing the function result.
    returns: None
    '''
    if event_args.function_result:
        # Extract token counts and model type
        token_counts = extract_token_counts_from_result(event_args.function_result)
        
        if token_counts:
            # Prepare and print the model type and token counts in a table format
            header = "| Token Type         | Count |"
            separator = "+--------------------+-------+"
            model_row = f"| Model Type         | {token_counts.pop('model_type', 'Unknown'):<5} |"
            
            print(separator)
            print(model_row)  # Print the model type row
            print(separator)
            print(header)
            print(separator)
            for token_type, count in token_counts.items():
                # Ensure we're not trying to print the model type again
                if token_type != 'model_type':
                    print(f"| {token_type:<18} | {count:>5} |")
            print(separator)
            
            # Here, you would send token_counts to your cost API
        else:
            print("Could not extract token counts from the function result.")


# Define a function to extract token counts from the function result
def extract_token_counts_from_result(function_result):
    '''
    This function extracts the used model and the token counts from the function result.
    args:
        - function_result: The result of the function invocation.
    returns: A dictionary containing the model type and token counts.
    '''

    # Ensure that the function_result object is not None
    if function_result is None:
        print("Function result is None.")
        return None
    
    # Access the inner content of the function result
    inner_content = function_result.get_inner_content()
    
    # Ensure inner_content is not None
    if inner_content is None:
        print("Inner content is None.")
        return None
    
    # Access the 'usage' attribute directly from the inner_content if it's structured as shown
    if hasattr(inner_content, 'usage'):
        usage_info = inner_content.usage
        
        # Extract token counts from the 'usage' attribute
        completion_tokens = getattr(usage_info, 'completion_tokens', 0)
        prompt_tokens = getattr(usage_info, 'prompt_tokens', 0)
        total_tokens = getattr(usage_info, 'total_tokens', 0)
        
        # Extract the model type from the inner_content
        model_type = getattr(inner_content, 'model', 'Unknown model')
        
        # Return the extracted token counts along with the model type
        return {
            'model_type': model_type,
            'completion_tokens': completion_tokens,
            'prompt_tokens': prompt_tokens,
            'total_tokens': total_tokens
        }
    else:
        print("Usage information is not available.")
        return None


# Keep Chatting 🤖
async def chat(input_text: str) -> None:
    # Save new message in the context variables
    print(f"User: {input_text}")
    chat_history.add_user_message(input_text)

    # Process the user message and get an answer
    answer = await kernel.invoke(chat_function, KernelArguments(user_input=input_text, history=chat_history))

    # Show the response
    print(f"ChatBot: {answer}")

    chat_history.add_assistant_message(str(answer))

# Assuming `kernel` is your instance of the semantic-kernel
# kernel.add_function_invoking_handler(pre_invocation_handler)
kernel.add_function_invoked_handler(post_invocation_handler)

# Let's define a prompt outlining a dialogue chat bot.
prompt = """
ChatBot can have a conversation with you about any topic.
It can give explicit instructions or say 'I don't know' if it does not have an answer.

{{$history}}
User: {{$user_input}}
ChatBot: """


# Register our semantic function
if selectedService == Service.OpenAI:
    execution_settings = sk_oai.OpenAIChatPromptExecutionSettings(
        service_id=service_id,
        ai_model_id="gpt-3.5-turbo-1106",
        max_tokens=2000,
        temperature=0.7,
    )
elif selectedService == Service.AzureOpenAI:
    execution_settings = sk_oai.OpenAIChatPromptExecutionSettings(
        service_id=service_id,
        ai_model_id=deployment,
        max_tokens=2000,
        temperature=0.7,
    )

prompt_template_config = sk.PromptTemplateConfig(
    template=prompt,
    name="chat",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(name="input", description="The user input", is_required=True),
        InputVariable(name="history", description="The conversation history", is_required=True),
    ],
    execution_settings=execution_settings,
)

chat_function = kernel.create_function_from_prompt(
    function_name="chat",
    plugin_name="chatPlugin",
    prompt_template_config=prompt_template_config,
)

# Create a chat history object and add a system message
chat_history = ChatHistory()
chat_history.add_system_message("You are a helpful chatbot who is good about giving book recommendations.")

# Start chatting !! 🚀
async def main():
    # Let's chat with the chatbot
    await chat("I'm looking for a book to read, do you have any suggestions?")

    # Add more chat function calls if needed
    await chat("I love history and philosophy, I'd like to learn something new about Greece, any suggestion?")

    # Chat chat chat!!!!    
    await chat("That sounds interesting, what is it about?")
    # await chat("if I read that book, what exactly will I learn about Greek history?")
    # await chat("could you list some more books I could read about this topic?")

    # print(chat_history)

# Start the script!
if __name__ == "__main__":
    asyncio.run(main())
