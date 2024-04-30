#이 프로그램은  긴 문장을 넣으면  다음 문장을 예상하는 프로그램으로
#긴 문장 단어 30개를 인풋으로 넣고  다음 문장 30개를 아웃풋검사에 사용하는 식으로
#프로그램이 돌아가면 어떤 아웃풋이 나올지 궁금해서 만들어 본다
#3개의 라이너를 통해 어떤 변화가 생길지 궁금하다

import torch
import torch.nn as nn

from io import open
import string

import re
import time
import random
import torchtext

from torchtext import data
import torch.optim as optim
from collections import Counter

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import requests


device ='cuda' if torch.cuda.is_available() else 'cpu'

# for reproducibility
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


#####

MAX_LENGTH = 100  #문장의 최대 길이
max_vocab_size=150000


optimizer_learning_rate  = 0.03  # modified learning rate from 0.01 to 1
drop_out_rate = 0.01

# 파라미터 설정
embedding_dim = 500
batch_size = 32
learning_rate = 0.002
num_epochs = 10000

sentence_length = 50



def readLangs():
    print("Reading lines...")

    response = requests.get('https://magu.co.kr/ai/book/news.htm')  #news.htm 잡다한 글 들 usserfall 어셔가의 몰락 책

    #파일을 읽고 그대로 사용


    return response



######################################################################
# 데이터 준비를 위한 전체 과정:
#
# -  텍스트 파일을 읽고 줄로 분리하고, 줄을 쌍으로 분리합니다.
# -  쌍을 이룬 문장들로 단어 리스트를 생성합니다.

response = readLangs()

def prepareData(response):

    # 문장을 단어로 분리
    if isinstance(response, str):
        words = response.strip()

    else:
        words = response.text.strip()


    print("Read %s words" % len(words))

    #문장을 단어로 분리
    tokens = re.findall(r'\b\w+\b', words)

    # 조사와 단어를 분리하는 정규식 패턴
    #pattern = r'(\w+)([은는을를가까까지])'
    #pattern = r'(\w+)(은|는|이|가|에서|까지)'

    # 정규식을 사용하여 조사와 단어 사이에 공백 추가
    #tokens = re.sub(pattern, r'\1 \2', words)

    #unique_tokens = list(set(tokens))  # 보케블러리에서 검색했을 때 원본 문장이 나오지 않는 문제로 삭제

    return tokens


inputs_split_data = prepareData(response)

print("inputs_data[5]", inputs_split_data[:10])



#########################################



# 어휘 구성
def build_vocabulary(data, max_vocab_size):
    counter = Counter()

    #tokens = re.findall(r'\b\w+\b', data)
    #unique_tokens = list(set(tokens))

    #print(data)

    counter.update(data)

    # 가장 많이 나타나는 단어들을 선택하여 어휘를 구성
    vocab = {word: idx for idx, (word, count) in enumerate(counter.most_common(max_vocab_size))}
    vocab['<PAD>'] = len(vocab)  # 패딩 토큰을 어휘에 추가
    vocab['<UNK>'] = len(vocab)

    vocab2text = {idx: word for word, idx in vocab.items()}

    return vocab,vocab2text


g_vocab, g_vocab2text = build_vocabulary(inputs_split_data, max_vocab_size)



print("len(g_vocab2text)",len(g_vocab2text))
print(g_vocab2text[0],g_vocab2text[1],g_vocab2text[2],g_vocab2text[3],g_vocab2text[4])


# 토큰화 및 어휘 인덱싱
def tokenize_and_index(data, vocab):

    #print("data[5]",data[5])
    #original_word = data[5]


    tokenized_data = []
    for sentence in data:
        indices = [vocab.get(sentence, vocab.get('<UNK>', len(vocab)))]
        tokenized_data.append(indices)

    print("tokenized_data",tokenized_data)
    return tokenized_data


    # 토큰화하고 어휘 인덱스에 맞게 변환
inputs_split_data_indexed = tokenize_and_index(inputs_split_data, g_vocab)


inputs_split_data_indexed_tensor = torch.tensor(inputs_split_data_indexed)


print ("inputs_split_data_indexed_tensor.size()", inputs_split_data_indexed_tensor.size())
# 텐서의 모양을 확인합니다.
shape = inputs_split_data_indexed_tensor.shape
print(f"텐서의 행 크기: {shape[0]}")
print(f"텐서의 열 크기: {shape[1]}")

reshaped_tensor = inputs_split_data_indexed_tensor.reshape((shape[1], shape[0]))

print ("inputs_split_data_indexed_tensor.size()", reshaped_tensor.size())

# 모델 정의

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


hidden_size = 1000
encoder1 = EncoderRNN(hidden_size, hidden_size).to(device)


class ChatbotModel(nn.Module):
    def __init__(self):
        super(ChatbotModel, self).__init__()
        self.embedding = nn.Embedding(len(g_vocab2text), embedding_dim)  #  임베딩 레이어

        self.elu = nn.ELU(alpha=1.0)
        self.dropout = nn.Dropout(drop_out_rate)
        self.output_layer1 = nn.Linear(embedding_dim, (embedding_dim*2))  #1번층


        self.output_layer2 = nn.Linear((embedding_dim*2), embedding_dim) #2번층


        self.output_layer3 = nn.Linear(embedding_dim, len(g_vocab2text)) #3번층
        # 출력 레이어:



    def forward(self, x):
        x = self.embedding(x)  # 임베딩 레이어를 통해 입력 처리
        x= self.elu(x)
        x = self.dropout(x)

        x = self.output_layer1(x)
        x= self.elu(x)

        x = self.output_layer2(x)
        x= self.elu(x)
        x = self.dropout(x)

        output = self.output_layer3(x)  # 출력 레이어를 통해 최종 출력 계산

        # 출력의 세 번째 차원을 평균 연산을 통해 축소합니다.
        # 예를 들어, 세 번째 차원(3124)에서 평균을 구해 `torch.Size([1, 3122])`로 변경합니다.
        #reduced_output = output.mean(dim=2)


        # (1, 3122, 3124) 모양의 텐서에서 최대값 계산하여 (1, 3122)로 축소
        # 세 번째 차원에 대한 최대값을 계산합니다.
        #output_tensor, _ = torch.max(output, dim=2)

        return output




# 모델 생성
model = ChatbotModel().to(device)

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 데이터셋을 미리 지정한 장치로 이동시킵니다.
inputs_split_data_indexed_tensor = reshaped_tensor.to(device)






# 문장에서 <pad> 토큰을 삭제하는 함수
def remove_pad_from_sentence(sentence, pad_token='<PAD>'):
    # str.replace() 메서드를 사용하여 <pad> 토큰을 빈 문자열로 대체하여 삭제
    cleaned_sentence = sentence.replace(pad_token, "")
    return cleaned_sentence.strip()  # 문장 양쪽 공백을 제거하여 깔끔한 문자열 반환

import time

print(inputs_split_data_indexed_tensor.size())
original_target_word = ""
for i in range(sentence_length-1):
    original_target_word = original_target_word + " "+  g_vocab2text[inputs_split_data_indexed_tensor[0,i].item()]
print("original_word",original_target_word)


print("len(inputs_split_data_indexed)%40",len(inputs_split_data_indexed)%sentence_length)

# 몫과 나머지 계산
quotient = len(inputs_split_data_indexed) // sentence_length
remainder = len(inputs_split_data_indexed) % sentence_length
print("quotient, remainder",quotient, remainder)


start_time = time.time()  # 시작 시간 기록

# 모델 훈련
for epoch in range(num_epochs):
    total_loss = 0



    # 배치로 하지 않고 문장의 길이로 처리  (어짜피 문장의 길이가 길어지면 처리 시간도 길어질테니까 ..

    for i in range(quotient-1):   # 남는 문장은 무시  ~ 450단어를 40단어로 나누면 25 :  25단어 무시..
                                  # 마지막 문장 -1은 모델의 예측에 넣어야 해서 뺌

        index = i * sentence_length         #0,40,80,120  (sentence_length가  40인경
        index2 = (sentence_length-1) +(i*sentence_length)  #39, 79,119,

        target_index = i * sentence_length +  sentence_length        #40,80,  (sentence_length가  40인경
        target_index2 = (sentence_length-1) +(i*sentence_length) + sentence_length  #79, 119,


        #모델에 들어갈 데이터와  값 비교할 데이터 추출
        input_subtensor = inputs_split_data_indexed_tensor[:, index:index2]  # 인덱스는 0부터 시작하므로 40번 인덱스부터 79번 인덱스까지 가져옴
        target_subtensor = inputs_split_data_indexed_tensor[:, target_index:target_index2]


        # 모델의 예측 수행
        #입력에는 첫번째 40 단어
        #출력에는 두번째 40 단어
        #print(input_subtensor)
        output_tensor = model(input_subtensor)


        target_subtensor = target_subtensor.squeeze(0)
        output_tensor = output_tensor.squeeze(0)
        #print(output_tensor.size())
        #print("target_subtensor",target_subtensor.size())


        loss = criterion(output_tensor.view(-1, len(g_vocab2text)), target_subtensor)


        total_loss += loss.item()

        # 옵티마이저 초기화 및 가중치 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



    # 각 에포크별 손실 출력
    if epoch % 50 == 0:

        print(f'Epoch {epoch + 1}, Loss: {total_loss :.4f}')

        end_time = time.time()  # 종료 시간 기록
        elapsed_time = end_time - start_time  # 총 학습 시간 계산
        print(f"학습 시간 {elapsed_time:.4f} 초")


        #_, predicted_index = torch.sort(output_tensor, dim=-1, descending=True)

        _, predicted_index = torch.max(output_tensor, dim=-1)

        #print("predicted_index   max ", predicted_index)

        predicted_word = ""
        for i in range(len(predicted_index)):
            predicted_word = predicted_word + " "+  g_vocab2text[predicted_index[i].item()]
        print("predicted_word",predicted_word)

        #가장 확률 높은 단어 추출
        #print("가장 확률 높은 단어 추출", output_tensor)

        #####################################



        #print(target_subtensor)  #원래의 글
        #print(target_subtensor[10].item())

        original_target_word = ""
        for i in range(sentence_length-1):
            original_target_word = original_target_word + " "+  g_vocab2text[target_subtensor[i].item()]
        print("original_target_word",original_target_word)


        ########################################
        #질문 하면 대답 듣는 걸로 한번 해보기
        #대홍수에 밀려내려가는 소리
        tempdata = "대홍수에 밀려내려가 는 소리"

        prepared_data = prepareData(tempdata)
        tensored_data = tokenize_and_index(prepared_data, g_vocab)
        tensored_data = torch.tensor(tensored_data).to(device)

        shape = tensored_data.shape
        tensored_data = tensored_data.reshape((shape[1], shape[0]))  #행,열로 모양 변경


        #print("tensored_data.size",tensored_data.size())



        outputed = model(tensored_data).to(device)
        outputed = outputed.squeeze(0)
        #outputed.view(-1, len(g_vocab2text))

        softmax_output = F.softmax(outputed, dim=-1)

        #print("output_tensor", output_tensor)
        # 결과 출력
        #print("softmax_output", softmax_output)

        #맥스 값이 이상하게 도 질문의 단어수에 맞춰서 출력된다  (질문단어 5개 답변 단어 5개 왜 ??)
        _, predicted_index = torch.max(softmax_output, dim=-1)
        print("predicted_index   max ", predicted_index)

        predicted_word = ""
        for i in range(len(predicted_index)):
            predicted_word = predicted_word + " "+  g_vocab2text[predicted_index[i].item()]

        print("대홍수에 밀려내려가는 소리에 대한 답변 ",predicted_word)



        """
        original_word = target_subtensor
        question_word = input_subtensor

        question_words =""
        for i in range(sentence_length):
            question_words = question_words + " "+  g_vocab2text[question_word[0, i].item()]
        print("input_word", remove_pad_from_sentence(question_words))

        original_words =""
        for i in range(sentence_length):
            original_words = original_words + " "+  g_vocab2text[original_word[0, i].item()]
        print("target_word", remove_pad_from_sentence(original_words))

        """

########################################
#모델 저장

import re, os
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

#모델 저장
model_save_path = os.path.join(save_dir, "book_preditc.pth")
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "g_vocab": g_vocab,  # 어휘 정보 저장
        "g_vocab2text": g_vocab2text  # 인덱스에서 단어로 변환하기 위한 사전 저장
    },
    model_save_path,
)
"""
########################################
#모델 로드
model2 = ChatbotModel().to(device)

model_load_path = os.path.join(save_dir, "book_preditc.pth")
checkpoint = torch.load(model_load_path)
model2.load_state_dict(checkpoint["model_state_dict"])

# 어휘 정보 로드
vocab = checkpoint["g_vocab"]
vocab2text_loaded = checkpoint["g_vocab2text"]


#tokenized_data= 'ㅋㅋ 마음 같아서는 대신 먹어주고 싶다 이거죠 .'
tokenized_data = '[1, 460, 461, 462, 463, 77, 464, 0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,]'

tokenized_data_list = eval(tokenized_data)

# 리스트를 파이토치 텐서로 변환
tokenized_data_tensor = torch.tensor(tokenized_data_list).to(device)

print(tokenized_data_tensor)


output_a = model2(tokenized_data_tensor).to(device)

print("output_a", output_a,output_a.size())


_, predicted_index = torch.max(output_a, dim=-1)


predicted_word =""

for i in range(sentence_length):
    predicted_word = predicted_word + " "+  vocab2text_loaded[predicted_index[i].item()]
print("answer", remove_pad_from_sentence(predicted_word))
"""
