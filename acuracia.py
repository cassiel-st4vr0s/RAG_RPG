import logging
from dotenv import load_dotenv
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from app import (
    get_conversation_chain,
    load_llm,
    get_vector_store,
    device,
)

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

EVAL_DATASET = {
    "questions": [
        "Um personagem pode conjurar uma magia de nível como ação bônus e outra como ação no mesmo turno?",
        "Quais são os efeitos da condição 'Amarrado' (Restrained)?",
        "Um Ladino pode usar Ataque Furtivo com uma espada grande de duas mãos?",
        "Como funciona um teste de resistência contra a morte (death saving throw)?",
        "Qual é a Classe de Armadura (CA) fornecida por uma Cota de Malha (Chain Mail)?",
        "A magia 'Nuvem de Adagas' (Cloud of Daggers) requer concentração?",
        "Quantas vezes um Bárbaro de 5º nível pode usar sua Fúria?",
        "Qual a diferença entre cobertura parcial (half cover) e cobertura de três quartos (three-quarters cover)?",
    ],
    "ground_truths": [
        "Não. Se um personagem conjura uma magia com tempo de conjuração de 1 ação bônus, a única outra magia que ele pode conjurar em seu turno é um truque (cantrip) com tempo de conjuração de 1 ação.",
        "Uma criatura amarrada tem sua velocidade reduzida a 0. Ataques contra ela têm vantagem, e os ataques da criatura têm desvantagem. A criatura também tem desvantagem em testes de resistência de Destreza.",
        "Não. A habilidade Ataque Furtivo requer que o Ladino use uma arma com a propriedade 'acuidade' (finesse) ou uma arma de ataque à distância. Uma espada grande não possui a propriedade 'acuidade'.",
        "No início de seu turno, um personagem com 0 pontos de vida faz um teste de resistência contra a morte rolando um d20. Um resultado de 10 ou mais é um sucesso, abaixo de 10 é uma falha. Três sucessos estabilizam o personagem; três falhas resultam em sua morte.",
        "Uma Cota de Malha fornece uma CA de 16 e impõe desvantagem em testes de Destreza (Furtividade).",
        "Sim, a magia 'Nuvem de Adagas' requer concentração para ser mantida.",
        "Um Bárbaro de 5º nível pode usar sua Fúria 3 vezes entre descansos longos.",
        "Cobertura parcial concede um bônus de +2 na CA e em testes de resistência de Destreza. Cobertura de três quartos concede um bônus de +5.",
    ]
}

def run_evaluation():
    load_dotenv()
    logging.info("Iniciando o script de avaliação...")

    # --- 1. Carregar componentes da aplicação RAG ---
    embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": device})
    vector_store = get_vector_store(text_chunks=[], embeddings=embeddings)
    generation_llm = load_llm() # Carrega o Gemini
    conversation_chain = get_conversation_chain(vector_store, generation_llm)

    # --- 2. Coletar respostas e contextos da aplicação ---
    logging.info("Coletando respostas do sistema RAG...")
    generated_answers = []
    retrieved_contexts = []
    for question in EVAL_DATASET["questions"]:
        response = conversation_chain.invoke({"question": question})
        generated_answers.append(response["answer"])
        contexts = [doc.page_content for doc in response["source_documents"]]
        retrieved_contexts.append(contexts)

    # --- 3. Preparar o dataset para o RAGAs ---
    dataset_dict = {
        "question": EVAL_DATASET["questions"],
        "answer": generated_answers,
        "contexts": retrieved_contexts,
        "ground_truth": EVAL_DATASET["ground_truths"],
    }
    eval_dataset = Dataset.from_dict(dataset_dict)

    # --- 4. Executar a avaliação usando Gemini como Juiz ---
    logging.info("Iniciando a avaliação com RAGAs usando Gemini...")
    
    # Criamos uma instância do Gemini para ser o "juiz"
    ragas_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    result = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm, # Passamos o Gemini como o LLM avaliador
        embeddings=embeddings, # Usamos o mesmo embedding da aplicação
    )

    logging.info("Avaliação concluída.")
    print("\n--- Resultados da Avaliação RAGAs (com Gemini) ---")
    print(result)
    print("---------------------------------------------------\n")

if __name__ == "__main__":
    run_evaluation()