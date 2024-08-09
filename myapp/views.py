from django.shortcuts import render
from langchain_core.output_parsers import JsonOutputParser
from langchain.chat_models import ChatOpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss, pickle, random
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

embedding = SentenceTransformer('jhgan/ko-sroberta-multitask')  # 임베딩 모델 가져오기
faiss_vectorstore = FAISS.load_local('./faiss_jiwoo', embedding, allow_dangerous_deserialization=True)

with open('./documents.pkl', 'rb') as file:
    documents = pickle.load(file)

query_set = """당신은 KB국민은행에서 제공하는 카드의 정보를 기반으로
                                         알맞은 카드를 추천해주도록 훈련된 AI입니다.
                                         1) 카드를 추천해주기 위한 정보 수집을 위해, 밸런스 질문을 만들어야합니다.
                                         2) 밸런스 질문이란, A와 B중 하나를 고르는 것을 말합니다.
                                         3) {num}개의 질문을 만듦니다.
                                         4) 일반적인 사람이 알아들을 수 있도록 쉬운 단어로 질문을 생성해야 합니다.
                                         5) 하나의 카드에 대한 문제가 아닌, 모든 카드를 기준으로 질문을 생성해야 합니다.
                                         6) 질문의 시작은 숫자나 줄바꿈, 여백 없이 질문만으로 답장합니다.
                                         7) 예문은 '저렴한 통신 요금 할인을 원합니다. 일상에서 다양한 할인 혜택을 원합니다.'입니다.
                                         8) Summary를 참조해서 만들어주세요.
                                         9) 정확한 추천을 위한 질문을 생성합니다
                                         10) 문제는 question에, 답은 ans에, 설명은 expl에 넣어주세요"""

def select_kind(request):
    return render(request, 'select_kind.html')

def select_card(request):
    return render(request, 'select_card.html')

def select_insurance(request):
    return render(request, 'select_insurance.html')

def select_deposit(request):
    return render(request, 'select_deposit.html')

def select_loan(request):
    return render(request, 'select_loan.html')

def select_count(request):
    return render(request, 'select_count.html')

class Retriever:
    @staticmethod
    def retrieve(cate, chap):

        cate_indices = []
        for doc_id, doc in faiss_vectorstore.docstore.items():
            if cate in doc.metadata.get('Cate', ''):
                cate_indices.append(doc_id)

        print(f"{cate} 카테고리에 속하는 문서 수: {len(cate_indices)}")

        # '금융' 카테고리에 속하는 문서들의 임베딩만 추출
        filtered_embeddings = np.array([faiss_vectorstore.index.reconstruct(doc_id) for doc_id in cate_indices])

        # 쿼리 임베딩 생성
        query_embedding = embedding.encode([chap]).reshape(1, -1)

        # '금융' 카테고리에 속하는 문서들 중에서 검색 수행
        D, I = faiss_vectorstore.index.search(query_embedding, len(filtered_embeddings))

        # 거리 D를 유사도로 변환
        similarities = 1 / (1 + D[0])

        # 유사도 기반으로 '금융' 카테고리 문서 필터링 및 결과 저장
        results = []
        for i in range(len(I[0])):
            if I[0][i] < len(cate_indices):
                doc_id = cate_indices[I[0][i]]  # 필터링된 인덱스에서 원래 문서 인덱스 가져오기
                doc = faiss_vectorstore.docstore[doc_id]
                similarity = similarities[i]

                results.append({
                    'score': similarity,
                    'content': doc.page_content,
                    'metadata': doc.metadata
                })

        print(f"최종 결과 문서 수: {len(results)}")

        random_results = random.sample(results, min(5, len(results)))

        results_dict = {'documents': []}
        # 랜덤으로 선택된 결과 출력
        for result in random_results:
            results_dict['documents'].append(result['content'])

        return results_dict

def make_qa(cate, chap, type_n):
    # 출력 파서 정의
    output_parser = JsonOutputParser(pydantic_object=query_set)  # 지정된 pydantic 모델에 맞게 데이터를 구조화하여 제공
    format_instructions = output_parser.get_format_instructions()
    query = query_set

    template = '''
        아래의 자료만을 사용하여 질문에 답하세요:
        {context}
        답변은 해당 형식에 맞게 모아서 만들어주세요:
        {form}

        질문: {question}
        '''

    model = ChatOpenAI(model="gpt-4o", temperature=0.5)

    # query = "수출입은행의 역할과 중요성 교육"
    results = Retriever.retrieve(cate, chap)
    prompt = PromptTemplate.from_template(template)

    chain = prompt | model | output_parser
    res = chain.invoke({'context': results['documents'], 'form': format_instructions, 'question': query})
    return res