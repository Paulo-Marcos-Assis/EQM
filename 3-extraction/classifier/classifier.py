import os
import joblib
import re
import string
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

class FraudClassifier:
    def __init__(self):
        # 1. Setup Paths
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.svm_path = os.path.join(self.base_path, 'auto_svm_best_model_bert.pkl')
        self.bert_path = os.path.join(self.base_path, 'bert_embeddings')  # Renamed folder earlier

        print("--- Initializing Fraud Classifier ---")

        # 2. Load SVM
        try:
            self.svm_model = joblib.load(self.svm_path)
            print("SVM Model loaded.")
        except Exception as e:
            print(f"Error loading SVM: {e}")
            raise e

        # 3. Load BERT
        try:
            self.bert_model = SentenceTransformer(self.bert_path)
            print("BERT Model loaded.")
        except Exception as e:
            print(f"Error loading BERT: {e}")
            raise e

    def _clean_text(self, text):
        """
        Applies the EXACT cleaning logic used in Jupyter Training.
        """
        # 1. Remove HTML
        text = BeautifulSoup(str(text), 'html.parser').get_text(separator=' ')
        
        # 2. Lowercase
        text = str(text).lower()
        
        # 3. Remove Punctuation
        text = re.sub(f'[{string.punctuation}]', '', text)
        
        # 4. Normalize Whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def predict(self, news_text):
        if not news_text:
            return None

        # STEP 1: CLEAN (Crucial Step!)
        cleaned_text = self._clean_text(news_text)
        
        # STEP 2: EMBED
        vector = self.bert_model.encode(cleaned_text)

        # STEP 3: RESHAPE
        vector_2d = vector.reshape(1, -1)

        # STEP 4: PREDICT
        prediction = self.svm_model.predict(vector_2d)
        
        return prediction[0]

# --- TEST AREA ---
if __name__ == "__main__":
    classifier = FraudClassifier()
    
    test_text = 'a 1ª vara da justiça federal de florianópolis prorrogou, nesta segunda-feira (10), o prazo a conclusão do inquérito da operação alcatraz para até 23 de novembro . deflagrada em 2019, a ação apura o supostorepasse de dinheiro público a servidores durante processos de licitação dentro do governo de santa catarina. a decisão é da juíza janaina cassol machado, que atendeu  a uma solicitação feita em junho pela polícia federal – responsável pelas investigações. o ministério público federal e as outras partes já foram notificadas da nova data. as informações são do portaljuscatarina. em 27 de junho, a polícia federal apresentou um relatório apontando indícios criminosos em umalicitação para a udesc(universidade do estado de santa catarina). segundo o órgão, neste mesmo texto foi solicitada a extensão do prazo para a conclusão das investigações. procurada nesta terça-feira (11), a pf informou que o pedido tem “relação a outros fatos em apuração”. com prazo de mais 120 dias, a investigação segue em segredo de justiça desde que foi deflagrada, em 30 de maio de 2019. naquela ocasião foram cumpridos 11 mandados de prisão (sendo sete preventivos e quatro temporários), e 41 mandados de busca e apreensão, em órgãos públicos, empresas e residências no estado. inicialmente, no mês de junho de 2019, a pf encaminhou à justiça federal três relatórios policiais ligados ao caso. no mês de agosto, mais outro. no mês de outubro, outros dois. e, ainda, mais cinco em novembro. já em 2020 foram encaminhados outros quatro relatórios policiais: um no mês de janeiro, outros dois no mês de março emais um em maioe outro em junho, totalizando 17 relatórios. em todos os relatórios, os indiciamentos foram realizados de acordo com as condutas praticadas por cada um dos investigados e indicam crimes de fraude à licitação e corrupção (ativa e passiva). leia também: operação alcatraz: grupo nd tem acesso exclusivo a primeira delação nesta manhã, a reportagem tentou contato com a trf4 (tribunal regional federal da 4ª região) para entender o que motivou a prorrogação do prazo. o ministério público federal também foi procurado. até às 11h30, no entanto, não houve retorno." Cleaned Input (What the model sees): a 1ª vara da justiça federal de florianópolis prorrogou nesta segundafeira 10 o prazo a conclusão do inquérito da operação alcatraz para até 23 de novembro deflagrada em 2019 a ação apura o supostorepasse de dinheiro público a servidores durante processos de licitação dentro do governo de santa catarina a decisão é da juíza janaina cassol machado que atendeu a uma solicitação feita em junho pela polícia federal – responsável pelas investigações o ministério público federal e as outras partes já foram notificadas da nova data as informações são do portaljuscatarina em 27 de junho a polícia federal apresentou um relatório apontando indícios criminosos em umalicitação para a udescuniversidade do estado de santa catarina segundo o órgão neste mesmo texto foi solicitada a extensão do prazo para a conclusão das investigações procurada nesta terçafeira 11 a pf informou que o pedido tem “relação a outros fatos em apuração” com prazo de mais 120 dias a investigação segue em segredo de justiça desde que foi deflagrada em 30 de maio de 2019 naquela ocasião foram cumpridos 11 mandados de prisão sendo sete preventivos e quatro temporários e 41 mandados de busca e apreensão em órgãos públicos empresas e residências no estado inicialmente no mês de junho de 2019 a pf encaminhou à justiça federal três relatórios policiais ligados ao caso no mês de agosto mais outro no mês de outubro outros dois e ainda mais cinco em novembro já em 2020 foram encaminhados outros quatro relatórios policiais um no mês de janeiro outros dois no mês de março emais um em maioe outro em junho totalizando 17 relatórios em todos os relatórios os indiciamentos foram realizados de acordo com as condutas praticadas por cada um dos investigados e indicam crimes de fraude à licitação e corrupção ativa e passiva leia também operação alcatraz grupo nd tem acesso exclusivo a primeira delação nesta manhã a reportagem tentou contato com a trf4 tribunal regional federal da 4ª região para entender o que motivou a prorrogação do prazo o ministério público federal também foi procurado até às 11h30 no entanto não houve retorno'
    
    print(f"\nRaw Input: {test_text}")
    
    cleaned = classifier._clean_text(test_text)
    print(f"Cleaned Input (What the model sees): {cleaned}")
    
    result = classifier.predict(test_text)
    print(f"Prediction Result: {result}")
    
    if result == 1:
        print("SUCCESS! It detected Fraud.")
    else:
        print("Still 0. The model might just think this specific sentence is safe.")
