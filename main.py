from chat import complete
import rag

chat_template = """<|im_start|>system
Speak using only language in the bible
You always answer using the following information
{}<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
"""


# Modify the main function as per requirements from README.md
# Below is a simple example of how to use the chat function

def main():

    while True:
        # Ask user for input and call the chat function
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break

        # Call the chat function with user input
        context = rag.search(user_input,10)

        content = chat_template.format('\n'.join(context), user_input)

        #print(content)
        response = complete(content )
 
        # Print the AI's response and exit
        print("AI:", response)

if __name__ == '__main__':
    # Launch the main() function if `python main.py` is entered from the terminal
    main()