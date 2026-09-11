import os
import json
import re
from typing import Dict, List, Optional
import httpx
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# ===========================
# MONKEY PATCH: Desabilita verificação SSL no httpx
# ===========================
# Necessário porque o Ollama usa certificados auto-assinados
_original_client_init = httpx.Client.__init__

def _patched_client_init(self, *args, **kwargs):
    kwargs['verify'] = False
    return _original_client_init(self, *args, **kwargs)

httpx.Client.__init__ = _patched_client_init

# ===========================
# CONFIGURAÇÕES (Carregadas de Var. de Ambiente ou Padrão)
# ===========================
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'https://ollama.ceos.ufsc.br')
SELECTED_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
LLM_TEMPERATURE = 0

# ===========================
# FUNÇÕES AUXILIARES
# ===========================
def normalize_edital(edital_str: Optional[str]) -> Optional[str]:
    """Remove zeros à esquerda do número do edital (ex: 005/2023 -> 5/2023)."""
    if edital_str is None:
        return None
    editorial = str(edital_str).strip()
    parts = editorial.split('/')
    if len(parts) == 2:
        num_norm = parts[0].lstrip('0')
        if not num_norm:
            num_norm = '0'
        return f"{num_norm}/{parts[1]}"
    else:
        return editorial

# ===========================
# CLASSE PRINCIPAL
# ===========================
class FeatureExtractor:
    def __init__(self):
        print(f"Feature Extractor configurado para usar Ollama em {OLLAMA_HOST} (modelo: {SELECTED_MODEL})")
        self.llm = None
        self._llm_initialized = False
    
    def _ensure_llm(self):
        """Lazy initialization do LLM - só conecta quando realmente precisar"""
        if not self._llm_initialized:
            print(f"Conectando ao Ollama em {OLLAMA_HOST}...")
            try:
                # Cria cliente Ollama com SSL desabilitado
                import ollama
                
                # O monkey patch do httpx já foi aplicado no import, então o cliente ollama
                # automaticamente usará httpx com verify=False
                
                self.llm = ChatOllama(
                    model=SELECTED_MODEL,
                    base_url=OLLAMA_HOST,
                    temperature=LLM_TEMPERATURE,
                    timeout=300,  # Aumentado para 5 minutos devido ao prompt longo
                    streaming=False,  # Desabilita streaming para evitar erro de parsing
                    client_kwargs={"verify": False}
                )
                self._llm_initialized = True
                print("Conexão com Ollama estabelecida com sucesso!")
            except Exception as e:
                print(f"ERRO ao conectar ao Ollama: {e}")
                import traceback
                traceback.print_exc()
                self.llm = None
                self._llm_initialized = True

    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Recebe o texto bruto da notícia e retorna o dicionário extraído.
        """
        default_return = {"municipio": [], "modalidade": [], "edital": [], "objeto": [], "unidades_gestoras": []}
        
        # Garante que o LLM está inicializado
        self._ensure_llm()
        
        if not self.llm:
            print("Erro: LLM não inicializado.")
            return default_return
        
        if not text or not isinstance(text, str):
            return default_return

        
        prompt_content = f"""
Você é um especialista em análise de notícias sobre licitações públicas.
Sua tarefa é identificar e extrair, de forma precisa e sem inferências, atributos específicos presentes **explicitamente** no texto.

Essas informações serão utilizadas posteriormente para cruzamento com bases públicas de licitações. Portanto, siga rigorosamente as regras de extração e normalização.

----------------------------------------------------------------------
INSTRUÇÕES GERAIS:
- Leia o todo o texto da notícia.
- Extraia SOMENTE informações que apareçam de forma explícita.
- Não faça inferências, não complete ausências e não reformule conteúdos.
- Para cada atributo, retorne uma lista (array). 
- Use [] quando não houver ocorrências.
- Remova duplicatas.
- A resposta deve ser APENAS um JSON válido.

----------------------------------------------------------------------
1) município:
- **O objetivo é encontrar o município relacionado à licitação**
- **Identifique e extraia o(s) município(s) que o texto da notícia aborda**.

Considerações:
- Se o texto mencionar vários municípios, mas um deles for claramente o principal, aquele que publicou o edital de licitação, então retorne apenas o município principal. 
- Se o texto mencionar diversos municípios e todos tiverem a mesma importância, retorne a lista com todos os municípios
- Se no texto não houver menção de nenhum município, não retorne nada (vazio que será preenchido por [])
- Se houver menção de municípios que não são do estado de Santa Catarina, não retorne nada (vazio que será preenchido por [])

**REGRA CRÍTICA - LICITAÇÕES ESTADUAIS:**

- **IMPORTANTE:** É possível que a licitação seja expedida pelo Governo do Estado de Santa Catarina (não por um município específico). Nesse caso, retorne "Estado de Santa Catarina"
- **ATENÇÃO:** Se o texto mencionar "Estado de Santa Catarina", "Governo do Estado", "Governo Estadual" ou órgãos estaduais (como "Secretaria de Estado da Saúde", "Secretaria de Estado da Administração"), isso indica uma licitação ESTADUAL, não municipal. Retorne "Estado de Santa Catarina"

Se o texto mencionar qualquer um dos seguintes, retorne "Estado de Santa Catarina":
1. "Estado de Santa Catarina" explicitamente
2. "Governo do Estado" ou "Governo Estadual"
3. Órgãos estaduais como "Secretaria de Estado da Saúde", "Secretaria de Estado da Administração", "Secretaria de Estado da Educação", etc.
4. Siglas de órgãos estaduais (SES, SEA, SED, SSP, etc.)




- Normalizações:
  - Remova expressões como "cidade de" ou "município de".
  - Mantenha apenas o nome principal.
  - Exemplo: "cidade de Florianópolis" → "Florianópolis"
  - Para licitações estaduais: "Governo do Estado" → "Estado de Santa Catarina"
  
### Exemplo few-shot:

Notícia 1:
"a polícia civil de santa catarina, por meio da 5ª delegacia especializada no combate à corrupção (5ª decor/chapecó), deflagrou uma operação na data de hoje, 02/05/2024, que resultou no cumprimento de nove mandados de busca e apreensão e três mandados de prisão nos municípios de quilombo/sc, são lourenço do oeste/sc e pato branco/pr.

a ação é um desdobramento da investigação de supostas fraudes em licitações no setor de obras do município de quilombo/sc, além de outras infrações penais correlatas, como formação de organização criminosa, lavagem de dinheiro e advocacia administrativa.
"
Note que: pelo eixo da fraude ser o municíio de quilombo/sc "*supostas fraudes em licitações no setor de obras do município de quilombo/sc*",a saída esperada nesse caso é:

Saída esperada:
{{
  "municipio_ente": "quilombo"
}}

Notícia 2:
"Operação mira 18 prefeituras de Santa Catarina por suspeita de fraude (...) a CNN apurou que 18 prefeituras do estado são alvos de busca e apreensão. São elas: São Miguel do Oeste, Guaraciaba, São José do Cedro, Bom Jesus do Oeste, Princesa, Bandeirantes, Flor do Sertão, São João do Oeste, Santa Helena, Sul Brasil, Descanso, Riqueza, Mondaí, Cordilheira Alta, Jardinópolis, Rio Fortuna, Águas Mornas e Antônio Carlos."

Saída esperada:
{{
  "municipio_ente": "São Miguel do Oeste, Guaraciaba, São José do Cedro, Bom Jesus do Oeste, Princesa, Bandeirantes, Flor do Sertão, São João do Oeste, Santa Helena, Sul Brasil, Descanso, Riqueza, Mondaí, Cordilheira Alta, Jardinópolis, Rio Fortuna, Águas Mornas e Antônio Carlos"
}}

Notícia 3 (LICITAÇÃO ESTADUAL - órgãos estaduais):
"O MPSC investiga irregularidades no pregão eletrônico 271/2023 destinado à contratação de empresa aérea especializada para locação de aeronaves. Interessados: Secretaria de Estado da Saúde – SES, Secretaria de Estado da Administração – SEA. O processo tramitou na Casa Militar e no Corpo de Bombeiros Militar de Santa Catarina."

Saída esperada:
{{
  "municipio_ente": "Estado de Santa Catarina"
}}

Note que: a presença de "Secretaria de Estado da Saúde", "Secretaria de Estado da Administração", indica que é uma licitação ESTADUAL, não municipal.

----------------------------------------------------------------------
2) modalidade:
Extraia a modalidade de licitação **somente se claramente identificada como modalidade**.

Modalidades válidas (exemplos):
- pregão presencial
- pregão eletrônico 
- concorrência
- concorrência presencial
- concorrência eletrônica
- convite
- tomada de preços
- dispensa de licitação
- inexigibilidade de licitação
- leilão
- concurso (apenas quando modalidade de licitação, não para concurso público)
- regime diferenciado de contratação (RDC)

REGRAS DE NORMALIZAÇÃO:
- Converter plurais para singular (ex: "pregões" → "pregão").
- "dispensa" (singular ou plural) → normalizar para "dispensa de licitação".
- "concorrência pública" → normalizar para apenas "concorrência".
- "concorrência" → só extraia quando for CERTEZA que refere-se à modalidade.

O QUE IGNORAR (NÃO EXTRAIR EM NENHUMA HIPÓTESE):
- “registro de preço” ou “sistema de registro de preço”.
- “pregão público” (termo genérico).
- “concorrência” usada no sentido de competição (ex: “frustrar a concorrência”).
- “concurso” usado para seleção de pessoal (ex: "concurso público para cargos").

----------------------------------------------------------------------
3) edital:
- Extraia TODOS os números de editais mencionados no texto, incluindo editais de diferentes modalidades (pregão, dispensa, inexigibilidade, tomada de preços, etc.).
- Se houver múltiplos editais no texto, extraia todos eles.
- Normalizar para formato "NUMERO/ANO", removendo prefixos como siglas de órgãos.
- Exemplos de normalização:
  - "Edital nº 123/2023" → "123/2023"
  - "pregão número 24 de 2022" → "24/2022"
  - "Dispensa de Licitação nº 283/SMLCP/SULIC/2023" → "283/2023"
  - "Pregão Eletrônico nº 058/SMLCP/SULIC/2024" → "58/2024"
  - "Tomada de Preços n. 03/2023" → "3/2023"
  - dispensa de licitação n. 0005/2020 → "5/2020"
- IMPORTANTE: Retorne TODOS os editais encontrados, mesmo que sejam de contexto histórico ou secundário.
- Se um edital for mencionado múltiplas vezes, inclua apenas uma vez (sem duplicatas).

----------------------------------------------------------------------
4) objeto:

**OBJETIVO:** Extrair a descrição COMPLETA do objeto da licitação, usando as PALAVRAS DA NOTÍCIA.

**CONTEXTO IMPORTANTE:**
As notícias descrevem licitações em linguagem jornalística — o texto da notícia pode não ser idêntico ao texto formal do edital, mas contém as mesmas informações. Sua tarefa é extrair o máximo de informação sobre o objeto que aparece na notícia a respeito de um determinado objeto (o que está sendo licitado), mesmo que seja uma descrição jornalística, paráfrase ou resumo.

**REGRAS CRÍTICAS:**
- Use as PALAVRAS DA NOTÍCIA, não invente ou substitua termos
- Se o objeto for descrito em vários trechos da notícia, combine-os em um único texto coerente
- Preserve obrigatoriamente:
  → A FINALIDADE da contratação (o que está sendo licitado)
  → O TIPO de objeto (serviço / obra / aquisição de material)
  → O CONTEXTO funcional (secretaria, órgão, local, beneficiário) quando mencionado
  → Especificações técnicas relevantes (marcas, modelos, quantidades) quando presentes
- Inclua o termo "registro de preços" quando mencionado
- NÃO inclua informações que não descrevem o objeto (ex: nome de investigados, datas do processo, valor pago)

**O QUE NÃO FAZER:**
- ❌ Reduzir a uma palavra ou frase genérica: "aquisição de materiais" em vez da descrição completa
- ❌ Inventar termos que não estão na notícia
- ❌ Trocar o texto da notícia pelo que seria o texto formal do edital
- ❌ Incluir contexto investigativo/jurídico que não descreve o objeto (ex: "conforme apurou a delegacia")

### EXEMPLOS FEW-SHOT:

**Exemplo 1 — A notícia descreve com linguagem jornalística:**
Notícia: "A Prefeitura de Campos Novos publicou pregão eletrônico para adquirir concreto asfáltico quente, material que será usado nos serviços de tapa-buraco nas ruas da cidade."

✅ CORRETO (usa as palavras da notícia, completo):
{{
  "objeto": ["adquirir concreto asfáltico quente para os serviços de tapa-buraco nas ruas da cidade"]
}}

❌ ERRADO (inventou texto do edital formal):
{{
  "objeto": ["REGISTRO DE PREÇO PARA AQUISIÇÃO DE CONCRETO ASFÁLTICO USINADO A QUENTE QUE SERÁ UTILIZADO PARA TAPA BURACOS EM VIAS PÚBLICAS"]
}}

❌ ERRADO (muito resumido):
{{
  "objeto": ["concreto asfáltico"]
}}

**Exemplo 2 — A notícia descreve com palavras próximas ao edital:**
Notícia: "A dispensa de licitação tinha como objetivo a contratação de empresa para aquisição de peças para a realização da manutenção dos implementos agrícolas da Secretaria de Agricultura."

✅ CORRETO:
{{
  "objeto": ["contratação de empresa para aquisição de peças para a realização da manutenção dos implementos agrícolas da Secretaria de Agricultura"]
}}

❌ ERRADO (resumido):
{{
  "objeto": ["compra de peças de reposição para maquinário agrícola"]
}}

**Exemplo 3 — Objeto com especificações técnicas detalhadas:**
Notícia: "O processo licitatório tinha como objeto a contratação de empresa, mediante dispensa de licitação, para prestação de serviço de seguro total dos equipamentos da usina de asfalto, por item, quais sejam: Usina de Asfalto Móvel 20-40 Ton/h, ano de fabricação 2018 Tanque bipartido - Diesel - Pesagem Ind. 380V; Vibroacabadora de Asfalto, marca CIBER, modelo AF 4000, sobre esteiras."

✅ CORRETO (incluir todos os detalhes técnicos):
{{
  "objeto": ["contratação de empresa para prestação de serviço de seguro total dos equipamentos da usina de asfalto, por item: Usina de Asfalto Móvel 20-40 Ton/h, ano de fabricação 2018 Tanque bipartido - Diesel - Pesagem Ind. 380V; Vibroacabadora de Asfalto, marca CIBER, modelo AF 4000, sobre esteiras"]
}}

❌ ERRADO (perdeu os detalhes técnicos):
{{
  "objeto": ["serviço de seguro dos equipamentos da usina de asfalto"]
}}

**Exemplo 4 — Registro de preços com beneficiários:**
Notícia: "O pregão tinha como objetivo registrar preços para o fornecimento parcelado de alimentos não perecíveis e itens correlatos para uso dos órgãos consorciados ao Consórcio Interfederativo Santa Catarina – CINCATARINA."

✅ CORRETO:
{{
  "objeto": ["registro de preços para fornecimento parcelado de alimentos não perecíveis e itens correlatos para uso dos órgãos consorciados ao Consórcio Interfederativo Santa Catarina – CINCATARINA"]
}}

❌ ERRADO (perdeu beneficiário):
{{
  "objeto": ["fornecimento de alimentos não perecíveis"]
}}

**Exemplo 5 — Serviços de TI com múltiplos componentes:**
Notícia: "A investigação revelou que a dispensa foi usada para contratar serviços de tecnologia da informação e comunicação, envolvendo acesso ao Diário Oficial dos Municípios de Santa Catarina, sistema de gestão de frotas e suporte técnico especializado."

✅ CORRETO (captura todos os componentes do serviço):
{{
  "objeto": ["contratação de serviços de tecnologia da informação e comunicação, envolvendo acesso ao Diário Oficial dos Municípios de Santa Catarina, sistema de gestão de frotas e suporte técnico especializado"]
}}

❌ ERRADO (resumiu demais):
{{
  "objeto": ["serviços de TI e comunicação"]
}}

**Exemplo 6 — Obras de infraestrutura:**
Notícia: "O pregão eletrônico foi aberto para contratar empresa especializada em pavimentação e drenagem de vias públicas no bairro Vila Nova, incluindo obras de meio-fio, sarjeta e calçadas, com área total de aproximadamente 12.000 m²."

✅ CORRETO:
{{
  "objeto": ["contratação de empresa especializada em pavimentação e drenagem de vias públicas no bairro Vila Nova, incluindo obras de meio-fio, sarjeta e calçadas, com área total de aproximadamente 12.000 m²"]
}}

❌ ERRADO:
{{
  "objeto": ["pavimentação de ruas"]
}}

**Exemplo 7 — Processo seletivo / concurso:**
Notícia: "A licitação foi aberta para contratar firma especializada em elaborar, organizar e aplicar processo seletivo simplificado, compreendendo todas as etapas — inscrição online, provas objetivas, correção e divulgação de resultados — para contratação temporária de servidores municipais."

✅ CORRETO:
{{
  "objeto": ["contratação de firma especializada para elaborar, organizar e aplicar processo seletivo simplificado, compreendendo todas as etapas: inscrição online, provas objetivas, correção e divulgação de resultados, para contratação temporária de servidores municipais"]
}}

❌ ERRADO:
{{
  "objeto": ["organização de processo seletivo"]
}}

**Exemplo 8 — Evitando excesso irrelevante:**

Notícia: "A prefeitura abriu licitação para aquisição de equipamentos de informática, incluindo computadores, monitores, teclados e mouses, destinados à modernização dos setores administrativos."

✅ CORRETO:
{{
  "objeto": ["aquisição de equipamentos de informática, incluindo computadores, monitores, teclados e mouses, destinados à modernização dos setores administrativos"]
}}

❌ ERRADO (excesso irrelevante inventado):
{{
  "objeto": ["aquisição de equipamentos de informática conforme padrões técnicos definidos em edital e normas administrativas internas"]
}}
**LEMBRE-SE:** Extraia o máximo de detalhes possível. É melhor ter texto demais do que texto de menos!

----------------------------------------------------------------------
5) unidades_gestoras:

**OBJETIVO:** Extrair nomes de órgãos, secretarias ou unidades gestoras mencionadas no texto.

**CONTEXTO IMPORTANTE:**
Este atributo é especialmente útil para licitações estaduais ou quando o município não é claramente identificado. Os nomes de secretarias e órgãos podem ser usados para cruzamento com o banco de dados.

**REGRAS:**
- Extraia nomes de secretarias estaduais (ex: "Secretaria de Estado da Saúde", "Secretaria de Estado da Administração")
- Extraia nomes de secretarias municipais (ex: "Secretaria Municipal de Educação")
- Extraia nomes de órgãos públicos (ex: "Departamento de Trânsito", "Fundação de Amparo à Pesquisa")
- Mantenha o nome completo como aparece no texto
- Normalize siglas conhecidas para o nome completo quando possível (ex: "SES" → "Secretaria de Estado da Saúde")
- Se não houver menção de órgãos/secretarias, retorne []

**EXEMPLOS:**

Notícia: "interessados: secretaria de estado da saúde – ses secretaria de estado da administração – sea"
✅ CORRETO:
{{
  "unidades_gestoras": ["Secretaria de Estado da Saúde", "Secretaria de Estado da Administração"]
}}

Notícia: "A Secretaria Municipal de Obras publicou edital..."
✅ CORRETO:
{{
  "unidades_gestoras": ["Secretaria Municipal de Obras"]
}}

----------------------------------------------------------------------
FORMATO DE RESPOSTA:
Retorne APENAS um JSON válido, no formato:

{{
  "municipio": [],
  "modalidade": [],
  "edital": [],
  "objeto": [],
  "unidades_gestoras": []
}}

**IMPORTANTE - ORDEM DE CORRESPONDÊNCIA:**
Quando houver múltiplos editais/modalidades/objetos, mantenha a ORDEM DE CORRESPONDÊNCIA entre eles:
- O primeiro item em "modalidade" deve corresponder ao primeiro item em "edital" e "objeto"
- O segundo item em "modalidade" deve corresponder ao segundo item em "edital" e "objeto"
- E assim sucessivamente

Exemplo:
Se a notícia menciona:
1. "Dispensa de Licitação nº 283/2023 para contratação de serviços de TI"
2. "Pregão Eletrônico nº 58/2024 para compra de software"

A saída deve ser:
{{
  "modalidade": ["dispensa de licitação", "pregão eletrônico"],
  "edital": ["283/2023", "58/2024"],
  "objeto": ["contratação de serviços de TI", "compra de software"]
}}

Se um atributo não tiver correspondente, use string vazia "" na posição correspondente para manter o alinhamento.

Texto da notícia:
\"\"\"{text}\"\"\"

Responda APENAS com o JSON válido, sem texto adicional.
"""
        # ==============================================================================

        try:
            # Invoca o LLM com uma única mensagem humana (seu prompt completo)
            print(f"[DEBUG] Tamanho do prompt: {len(prompt_content)} caracteres")
            print(f"[DEBUG] Tamanho do texto da notícia: {len(text)} caracteres")
            print("[DEBUG] Iniciando chamada ao LLM...")
            
            response = self.llm.invoke([HumanMessage(content=prompt_content)])
            
            print("[DEBUG] LLM respondeu! Processando resposta...")
            result = response.content.strip()
            print(f"[DEBUG] Resposta da LLM (primeiros 500 chars): {result[:500]}")
            print(f"[DEBUG] Tamanho total da resposta: {len(result)} caracteres")
            
            return self._parse_json_response(result, default_return)

        except Exception as e:
            # Em produção, logamos o erro mas não paramos o pipeline
            print(f"[Erro na Extração] {e}")
            import traceback
            traceback.print_exc()
            return default_return

    def _parse_json_response(self, result_str: str, default_return: Dict) -> Dict:
        """
        Limpa e valida o JSON retornado pelo modelo.
        """
        # Limpeza de blocos de código Markdown (comum em LLMs)
        if result_str.startswith("```json"):
            result_str = result_str[7:]
        if result_str.startswith("```"):
            result_str = result_str[3:]
        if result_str.endswith("```"):
            result_str = result_str[:-3]
        
        result_str = result_str.strip()

        try:
            data = json.loads(result_str)
            
            # Garante a estrutura de saída
            out = {}
            for key in ["municipio", "modalidade", "edital", "objeto", "unidades_gestoras"]:
                value = data.get(key, [])
                
                # Força formato de lista de strings
                if isinstance(value, str):
                    value = [value] if value.strip() else []
                elif isinstance(value, (int, float)):
                    value = [str(value)]
                elif not isinstance(value, list):
                    value = []

                # Remove duplicatas e strings vazias
                clean_list = []
                seen = set()
                for item in value:
                    s = str(item).strip()
                    # Remove aspas duplas ou simples no início e fim
                    s = s.strip('"').strip("'").strip()
                    if s and s not in seen:
                        clean_list.append(s)
                        seen.add(s)
                
                out[key] = clean_list

            # Aplica normalização específica de editais
            out['edital'] = [normalize_edital(e) for e in out.get('edital', []) if e]

            return out

        except json.JSONDecodeError:
            print(f"Falha ao decodificar JSON. Início da resposta: {result_str[:50]}...")
            return default_return


