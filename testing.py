""" import os
from langchain.chat_models import ChatOpenAI
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

LLM = ChatOpenAI(max_tokens=1000, temperature=0.2)
MEMORY = ConversationBufferMemory(memory_key="chat_history")
LLM_CHAIN = LLMChain(llm=LLM)
AGENT = ZeroShotAgent(llm_chain=LLM_CHAIN, verbose=True)
AGENT_CHAIN = AgentExecutor.from_agent_and_tools(
    agent=AGENT, verbose=True, memory=MEMORY
)

try:
    answer = AGENT_CHAIN.run("Where do apples grow from?")
    print(answer)
    
except Exception as e:
    answer = str(e)
    print(answer)
 """

from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are an information extractor. For every phrase i give you, please extract the main key words to better find the desired movie."},
    {"role": "user", "content": "I want to find a movie about turtles where Jason Statham plays the main role"}
  ]
)

print(completion.choices[0].message.content)





"""

tratar dataset

treinar modelo

LLM que pega no score previsto e na descrição do jogo, e tenta criar uma lista de pontos fortes e fracos






"""