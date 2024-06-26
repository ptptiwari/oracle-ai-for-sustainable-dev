# Integrate OCI Generative AI with LangChain
This repo contains all the work done to develop demos on the integration between [**LangChain**](https://www.langchain.com/) and Oracle [**OCI GenAI**](https://www.oracle.com/artificial-intelligence/generative-ai/large-language-models/) Service.

## OCI Generative AI Service is in GENERAL AVAILABILITY
Consider that OCI Generative AI Service (based on Cohere models) is now (July ) 

## Oracle 23ai is in GENERAL AVAILABILITY
Consider that Oracle 23 ai database is available now

## Documentation
The development of the proposed integration is based on the example, from LangChain, provided [here](https://python.langchain.com/docs/modules/model_io/models/llms/custom_llm)

**RAG** has been first described in the following [arXiv paper](https://arxiv.org/pdf/2005.11401.pdf)

## Features
* How-to build a complete, end-2-end RAG solution using LangChain and Oracle GenAI Service.
* How-to load multiple pdf
* How-to split pdf pages in smaller chuncks
* How-to do semantic search using Embeddings
* How-to use Cohere Embeddings
* How-to use HF Embeddings
* How-to setup a Retriever using Embeddings
* How-to add Cohere reranker to the chain
* How to integrate OCI GenAI Service with LangChain
* How to define the LangChain
* How to use the Oracle vector Db capabilities
* How to use in-memory database capability

## Oracle BOT
Using the script [run_oracle_bot_exp.sh](./run_oracle_bot_exp.sh) you can launch a simple ChatBot that showcase Oracle GenAI service. The demo is based on docs from Oracle Database pdf documentation.

You need to put in the local directory:
* Trobleshooting.pdf
* globally-distributed-autonomous-database.pdf
* Oracle True cache.pdf
* oracle-database-23c.pdf
* oracle-globally-distributed-database-guide.pdf
* sharding-adg-addshard-cookbook-3610618.pdf

You can add more pdf. Edit [config_rag.py](./config_rag.py)

## Video





