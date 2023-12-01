import openai
import time
from dotenv import load_dotenv
import os

try:
    load_dotenv()  # take environment variables from .env.
except Exception as oops:
    print("Issue with load_dotenv:" + oops)

openai.api_key = os.getenv("OPENAI_API_KEY")

assistant_id = "asst_ITK3pwrDTlOduKpL93sUea97"

def create_thread(ass_id,prompt):
    #Get Assitant
    assistant = openai.beta.assistants.retrieve(ass_id)

    #create a thread
    thread = openai.beta.threads.create()
    my_thread_id = thread.id


    #create a message
    message = openai.beta.threads.messages.create(
        thread_id=my_thread_id,
        role="user",
        content=prompt
    )

    #run
    run = openai.beta.threads.runs.create(
        thread_id=my_thread_id,
        assistant_id=ass_id,
    ) 

    return run.id, thread.id


def check_status(run_id,thread_id):
    run = openai.beta.threads.runs.retrieve(
        thread_id=thread_id,
        run_id=run_id,
    )
    return run.status


my_run_id, my_thread_id = create_thread(assistant_id,"[topic]: fraud conversation")


status = check_status(my_run_id,my_thread_id)

while (status != "completed"):
    status = check_status(my_run_id,my_thread_id)
    time.sleep(1)


response = openai.beta.threads.messages.list(
  thread_id=my_thread_id
)


if response.data:
    print(response.data[0].content[0].text.value)