import os
import time
import json
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urljoin
from pathlib import Path
from service_essentials.basic_service.cached_collector_service import CachedCollectorService
from tools import parse_iso_or_portuguese_date, parse_dmy_format, parse_other_format, parse_g1_format
from service_essentials.object_storage_manager.minio_manager import MinIOManager  


# Mapeamento de funções de parsing de data
DATE_PARSERS = {
    'iso_or_portuguese': parse_iso_or_portuguese_date,
    'dmy': parse_dmy_format,
    'other_format': parse_other_format,
    'g1_format': parse_g1_format,
}

# ==============================================================================
# CLASSE PRINCIPAL DO CRAWLER
# ==============================================================================

class CollectorNoticias(CachedCollectorService):
    def __init__(self, config_file='crawler_configs.json'):
        super().__init__(data_source='noticias')  # Initialize the parent class (BasicProducerConsumerService)
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                all_configs = json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Arquivo de configuração '{config_file}' não encontrado.")
            raise
        except json.JSONDecodeError:
            self.logger.error(f"Erro ao decodificar o arquivo JSON de configuração '{config_file}'.")
            raise

        self.aux = all_configs
        self.config = None
        self.portal_name = None
        self.target_date = None
        self.base_url_template = None
        self.page_1_url = None
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.date_parser = None
        self.min_page = 1
        self.max_page = None
        
        # Diretório local para salvar JSONs das notícias
        self.local_download_dir = Path(__file__).parent / "downloaded_news"
        self.local_download_dir.mkdir(exist_ok=True)

    def collect_data(self, message):
        if message.get("folder_path"):
            folder_path = message.get("folder_path")
            self.portal_name = message.get("portal_name", "local_folder")
            entity_type = message.get("entity_type")

            if folder_path.startswith("/"):
                articles = self.process_local_filesystem(folder_path)
            else:
                articles = self.process_local_folder(folder_path)

            if entity_type and articles:
                for article in articles:
                    article["entity_type"] = entity_type
            return articles

        # Suporte para coleta completa de qualquer portal
        if message.get("collect_all") == "yes":
            self.portal_name = message.get("portal_name")
            self.base_url_template = message.get("url")
            self.page_1_url = (message.get("parameters") or {}).get("page_1_url")
            entity_type = message.get("entity_type")

            if self.portal_name == "nsc":
                articles = self.collect_all_nsc()
            else:
                articles = self.collect_all_portal()

            if entity_type and articles:
                for article in articles:
                    article["entity_type"] = entity_type
            return articles

        # Manter compatibilidade com collect_all_nsc antigo
        if message.get("collect_all_nsc") == "yes" and message.get("portal_name") == "nsc":
            self.portal_name = message.get("portal_name", "local_folder")
            self.base_url_template = message.get("url")
            self.page_1_url = (message.get("parameters") or {}).get("page_1_url")
            entity_type = message.get("entity_type")
            articles = self.collect_all_nsc()
            if entity_type and articles:
                for article in articles:
                    article["entity_type"] = entity_type
            return articles

        self.logger.debug(f"processando mensagem: {message}")
        parameters = message.get("parameters") or {}
        self.portal_name = parameters.get("portal")
        self.base_url_template = message.get("url")
        self.page_1_url = parameters.get("page_1_url")
        start_date_str = message.get("start_date")
        entity_type = message.get("entity_type")

        if not self.portal_name:
            self.logger.error("Mensagem sem parameters.portal")
            return

        if not start_date_str:
            self.logger.error("Mensagem sem campo start_date")
            return

        try:
            self.target_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        except Exception as e:
            self.logger.error(f"Erro ao interpretar start_date '{start_date_str}': {e}")
            return

        self.config = self.aux.get(self.portal_name)
        if not self.config:
            self.logger.error(f"Configuração não encontrada para portal '{self.portal_name}'")
            return

        self.date_parser = DATE_PARSERS[self.config['date_parser']]
        
        # Usar max_page da configuração se disponível, senão descobrir automaticamente
        if 'max_page' in self.config and self.config['max_page']:
            self.max_page = self.config['max_page']
            self.logger.info(f"Usando max_page da configuração: {self.max_page}")
        else:
            self.max_page = self._find_max_page()
        
        # Usar min_page da configuração se disponível
        if 'min_page' in self.config and self.config['min_page']:
            self.min_page = self.config['min_page']
        
        bucket = os.getenv("PUBLIC_BUCKET", "workflow-hmg")

        print(f"[INFO] Iniciando crawler para o portal '{self.portal_name}' na data {self.target_date}", flush=True)
        self.logger.info(f"Iniciando crawler para o portal '{self.portal_name}' na data {self.target_date}")
        
        # Verificar se a data alvo é recente (últimos 7 dias)
        today = datetime.now().date()
        days_ago = (today - self.target_date).days
        is_recent_date = days_ago <= 7
        
        # Se max_page = 1, coletar apenas da página 1 sem busca binária
        if self.max_page == 1:
            print("[INFO] Portal configurado para coletar apenas da página 1", flush=True)
            target_pages = [1]
        # Se skip_binary_search = true OU se a data for recente (Day-1 ou Day-2), coletar sequencialmente
        elif self.config.get('skip_binary_search', False) or is_recent_date:
            if is_recent_date and not self.config.get('skip_binary_search', False):
                print(f"[INFO] Data recente detectada (Day-{days_ago}). Usando coleta sequencial das primeiras páginas.", flush=True)
                # Para datas recentes sem configuração explícita, usar no máximo 10 páginas
                max_pages_to_collect = min(self.max_page, 10) if self.max_page else 10
                target_pages = list(range(self.min_page, max_pages_to_collect + 1))
            else:
                print(f"[INFO] Portal configurado para coletar todas as páginas de {self.min_page} a {self.max_page}", flush=True)
                target_pages = list(range(self.min_page, self.max_page + 1))
        else:
            print(f"[INFO] Data antiga (Day-{days_ago}). Usando busca binária.", flush=True)
            candidate_page = self._binary_search_for_date_page()
            if candidate_page is None:
                self.logger.error("Não foi possível encontrar uma página com a data alvo. Encerrando.")
                return
            target_pages = self._find_all_target_pages(candidate_page)
        if not target_pages:
            self.logger.error("Nenhuma página confirmada para a data alvo. Encerrando.")
            return
            
        print(f"[INFO] Páginas a serem processadas: {target_pages}", flush=True)
        self.logger.info(f"Páginas a serem processadas: {target_pages}")
        
        all_articles_data = []
        article_urls_processed = set()
        
        for page_num in tqdm(target_pages, desc="Coletando artigos das páginas"):
            page_url = self._get_page_url(page_num)
            soup = self._get_soup(page_url)
            if not soup: continue
            
            links = self._get_article_links(soup)
            print(f"[INFO] Página {page_num}: {len(links)} artigos encontrados", flush=True)
            for url in tqdm(links, desc=f"Artigos da página {page_num}", leave=False):
                if url in article_urls_processed:
                    continue
                
                article_date = self._extract_article_date(url)
                
                if article_date == self.target_date:
                    article_info = self._extract_article_info(url, page_num)
                    if article_info:
                        all_articles_data.append(article_info)
                        print(f"[✓] Artigo coletado: {article_info.get('title', 'sem título')[:60]}...", flush=True)
                        
                article_urls_processed.add(url)
                time.sleep(0.2)
        
        if not all_articles_data:
            self.logger.warning("Nenhum artigo encontrado para a data especificada após verificação.")
            return

        # Add entity_type to each article if provided in the message
        if entity_type:
            for article in all_articles_data:
                article["entity_type"] = entity_type

        output_dir = "collected_articles"
        os.makedirs(output_dir, exist_ok=True)
        file_date = self.target_date.strftime("%d-%m-%y")
        output_filename = f"{self.portal_name}_articles_{file_date}.json"
        output_path = os.path.join(output_dir, output_filename)

        print(f"[INFO] Salvando {len(all_articles_data)} artigos em: {output_path}", flush=True)
        self.logger.info(f"Salvando {len(all_articles_data)} artigos em: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_articles_data, f, ensure_ascii=False, indent=4)
        
        print("[INFO] Coleta concluída com sucesso!", flush=True)
        self.logger.info("Coleta concluída com sucesso!")
        return all_articles_data

    def _get_page_url(self, page_num):
        """Retorna a URL correta para uma página, considerando URLs especiais."""
        if self.max_page == 1:
            return self.base_url_template
        if page_num == 1 and self.page_1_url:
            return self.page_1_url
        return self.base_url_template.format(page_num)

    def _get_soup(self, url):
        """Busca o conteúdo de uma URL e retorna um objeto BeautifulSoup."""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Falha ao buscar a URL {url}: {e}")
            return None

    def _find_max_page(self):
        """Encontra o número máximo de páginas usando busca exponencial e binária."""
        print("[INFO] Iniciando busca automática pelo número máximo de páginas...", flush=True)
        self.logger.info("Iniciando busca automática pelo número máximo de páginas...")
        test_page = 1
        while True:
            url = self._get_page_url(test_page)
            self.logger.info(f"Busca exponencial: testando página {test_page}... URL: {url}")
            soup = self._get_soup(url)
            if not soup:
                self.logger.warning(f"Página {test_page}: Falha ao obter soup")
                break
            links = self._get_article_links(soup)
            self.logger.info(f"Página {test_page}: {len(links)} links encontrados")
            if not links:
                self.logger.warning(f"Página {test_page}: Nenhum link encontrado, parando busca exponencial")
                break
            test_page *= 2

        left, right = test_page // 2, test_page
        max_page = left
        print(f"[INFO] Iniciando busca binária no intervalo [{left}, {right}]", flush=True)
        self.logger.info(f"Iniciando busca binária no intervalo [{left}, {right}]")
        while left <= right:
            mid = (left + right) // 2
            if mid == 0: break
            url = self._get_page_url(mid)
            self.logger.info(f"Busca binária: testando página {mid} no intervalo {left}-{right}... URL: {url}")
            soup = self._get_soup(url)
            if soup:
                links = self._get_article_links(soup)
                self.logger.info(f"Página {mid}: {len(links)} links encontrados")
                if links:
                    max_page = mid
                    left = mid + 1
                else:
                    right = mid - 1
            else:
                self.logger.warning(f"Página {mid}: Falha ao obter soup")
                right = mid - 1
        
        print(f"[INFO] Número máximo de páginas encontrado: {max_page}", flush=True)
        self.logger.info(f"Número máximo de páginas encontrado: {max_page}")
        return max_page

    def _get_article_links(self, soup):
        """Extrai links de artigos da página com base na configuração do portal."""
        links = set()
        finder = self.config['link_finder']
        containers = soup.find_all(finder['container']['tag'], finder['container'].get('attrs', {})) if finder.get('container') else [soup]

        self.logger.debug(f"_get_article_links: Procurando por tag='{finder['tag']}' com attrs={finder.get('attrs', {})}")

        for container in containers:
            elements = container.find_all(finder['tag'], finder.get('attrs', {}))
            self.logger.debug(f"_get_article_links: Encontrados {len(elements)} elementos")

            for element in elements:
                href = element.get('href')
                if not href:
                    continue
                href = urljoin(self.base_url_template, href)
                href = href.split('?')[0].split('#')[0]

                # Check exclusions first (blacklist)
                if 'href_excludes' in finder:
                    if any(exclude in href for exclude in finder['href_excludes']):
                        self.logger.debug(f"Link rejeitado (excluído): {href}")
                        continue

                # Then check inclusions (whitelist) if specified
                if 'href_prefixes' in finder:
                    if any(href.startswith(prefix) for prefix in finder['href_prefixes']):
                        # pular se for exatamente a URL da listagem/categoria (não é artigo)
                        if any(href == prefix for prefix in finder['href_prefixes']):
                            self.logger.debug(f"Link rejeitado (é a própria listagem): {href}")
                            continue
                        links.add(href)
                        self.logger.debug(f"Link aceito (com prefix): {href}")
                    else:
                        self.logger.debug(f"Link rejeitado (sem prefix válido): {href}")
                else:
                    links.add(href)
                    self.logger.debug(f"Link aceito (sem filtro de prefix): {href}")

        self.logger.debug(f"_get_article_links: Total de {len(links)} links únicos extraídos")
        return list(links)

    def _extract_article_date(self, url):
        """Extrai a data de publicação de um único artigo."""
        soup = self._get_soup(url)
        if not soup:
            return None
        date_selectors = self.config['article_selectors']['date']
        if not isinstance(date_selectors, list):
            date_selectors = [date_selectors]
        for selector in date_selectors:
            # Se não há attrs específicos, buscar TODOS os elementos da tag e tentar parse
            if not selector.get('attrs'):
                tags = soup.find_all(selector['tag'])
                for tag in tags:
                    date_str = tag.get(selector.get('attribute', '')) or tag.get_text(strip=True)
                    if date_str:
                        try:
                            parsed_date = self.date_parser(date_str)
                            if parsed_date:  # Se conseguiu fazer parse, é uma data válida
                                return parsed_date
                        except:
                            continue  # Não é uma data válida, tentar próximo elemento
            else:
                # Com attrs específicos, usar find normal
                tag = soup.find(selector['tag'], selector.get('attrs', {}))
                if tag:
                    date_str = tag.get(selector.get('attribute', '')) or tag.get_text(strip=True)
                    if date_str:
                        return self.date_parser(date_str)
        self.logger.warning(f"Nenhuma data encontrada para o artigo: {url}")
        return None

    def _check_page_for_target_date(self, page_num):
        """Verifica se uma página contém artigos da data alvo e retorna detalhes."""
        page_url = self._get_page_url(page_num)
        soup = self._get_soup(page_url)
        if not soup: return {"has_target": False, "first_date": None, "last_date": None}
        links = self._get_article_links(soup)
        if not links: return {"has_target": False, "first_date": None, "last_date": None}
        
        article_dates = []
        for link in links:
            date = self._extract_article_date(link)
            if date:
                article_dates.append(date)

        if not article_dates: return {"has_target": False, "first_date": None, "last_date": None}

        first_date = max(article_dates)
        last_date = min(article_dates)

        has_target = self.target_date in article_dates
        
        return {"has_target": has_target, "first_date": first_date, "last_date": last_date}

    def _binary_search_for_date_page(self):
        """Usa busca binária para encontrar uma página que contenha a data alvo."""
        print(f"[INFO] Iniciando busca binária pela data {self.target_date}...", flush=True)
        self.logger.info(f"Iniciando busca binária pela data {self.target_date}...")
        left, right = self.min_page, self.max_page
        with tqdm(total=(right - left), desc="Busca binária por data") as pbar:
            while left <= right:
                mid = (left + right) // 2
                if mid == 0: break
                # pbar.set_description(f"Verificando página {mid}")
                # pbar.update(1)
                page_info = self._check_page_for_target_date(mid)
                first_date, last_date = page_info['first_date'], page_info['last_date']
                self.logger.info(f"Página {mid}: Data mais recente={first_date}, Data mais antiga={last_date}")
                if page_info['has_target']:
                    print(f"[INFO] Data alvo encontrada na página {mid}!", flush=True)
                    self.logger.info(f"Data alvo encontrada na página {mid}!")
                    return mid
                if first_date and first_date > self.target_date:
                    left = mid + 1
                elif last_date and last_date < self.target_date:
                    right = mid - 1
                else:
                    right = mid - 1
                time.sleep(0.5)
        self.logger.warning("Nenhuma página candidata encontrada na busca binária.")
        return None

    def _find_all_target_pages(self, start_page):
        """A partir de uma página inicial, encontra todas as páginas vizinhas com a data alvo."""
        if start_page is None: return []
        self.logger.info(f"Verificando vizinhança da página {start_page}...")
        target_pages = {start_page}
        for page in range(start_page - 1, self.min_page - 1, -1):
            self.logger.info(f"Verificando página anterior: {page}")
            page_info = self._check_page_for_target_date(page)
            if page_info["has_target"]:
                target_pages.add(page)
            # elif page_info["last_date"] and page_info["last_date"] > self.target_date:
            #     continue
            else:
                break
            time.sleep(0.5)
        for page in range(start_page + 1, self.max_page + 1):
            self.logger.info(f"Verificando página seguinte: {page}")
            page_info = self._check_page_for_target_date(page)
            if page_info["has_target"]:
                target_pages.add(page)
            # elif page_info["first_date"] and page_info["first_date"] < self.target_date:
            #     break
            else:
                break
            time.sleep(0.5)
        return sorted(list(target_pages))

    def _extract_article_info(self, url, page_num):
        """Extrai todas as informações de um artigo e retorna um dicionário."""
        soup = self._get_soup(url)
        if not soup: return None
        selectors = self.config['article_selectors']
    
        # 1. Title
        title_tag = soup.find(selectors['title']['tag'], selectors['title'].get('attrs', {}))
        title = title_tag.get_text(strip=True) if title_tag else "sem-titulo"
        
        # 2. Subtitle (Chamada) - NEW LOGIC
        chamada = None
        subtitle_tag = None
        if 'subtitle' in selectors:
            subtitle_selector = selectors['subtitle']
            subtitle_tag = soup.find(subtitle_selector['tag'], subtitle_selector.get('attrs', {}))
            chamada = subtitle_tag.get_text(strip=True) if subtitle_tag else None
            
            # Log apenas quando encontrar ou não encontrar a chamada
            if chamada:
                self.logger.info(f"✅ Chamada extraída: {chamada[:80]}...")
            else:
                self.logger.warning(f"⚠️  Chamada não encontrada para: {url}")


        # 3. Content
        text = ""
        content_selector = selectors['content']
        content_container = soup.find(content_selector['tag'], content_selector.get('attrs', {})) if 'tag' in content_selector else soup
        if content_container:
            elements = content_container.find_all(content_selector['find_all'])
            text = "\n".join(el.get_text(strip=True) for el in elements)

        # 4. Date
        extracted_date = self._extract_article_date(url)
        date_publication = extracted_date.isoformat() if extracted_date else None

        article_data = {
            "portal": self.portal_name,
            "title": title,
            "chamada": chamada,  # ✅ FIXED
            "text": text,
            "url": url,
            "date_publication": date_publication,
            "date_extraction": datetime.utcnow().isoformat(),
            "page_number": page_num
        }
        
        # Log do JSON completo para validação
        self.logger.info("="*80)
        self.logger.info(f"📰 NOTÍCIA EXTRAÍDA - {self.portal_name.upper()}")
        self.logger.info("-"*80)
        self.logger.info(f"Portal: {article_data['portal']}")
        self.logger.info(f"Título: {article_data['title'][:100]}...")
        self.logger.info(f"Chamada: {article_data['chamada'][:100] if article_data['chamada'] else '❌ VAZIO'}")
        self.logger.info(f"Conteúdo: {len(article_data['text'])} caracteres")
        self.logger.info(f"URL: {article_data['url']}")
        self.logger.info(f"Data Publicação: {article_data['date_publication'] or '❌ VAZIO'}")
        self.logger.info(f"Data Extração: {article_data['date_extraction']}")
        self.logger.info(f"Página: {article_data['page_number']}")
        self.logger.info("="*80)
        
        return article_data

    def process_local_filesystem(self, folder_path):
        """
        Lê todos os arquivos JSON de uma pasta do filesystem local.
        Args:
            folder_path: Caminho absoluto no filesystem (ex: /app/local_news)
        Returns:
            Lista de todos os artigos carregados dos arquivos JSON.
        """
        self.logger.info(f"Iniciando leitura dos arquivos JSON do filesystem em: {folder_path}")

        if not os.path.exists(folder_path):
            self.logger.error(f"Diretório não encontrado: {folder_path}")
            return []
        
        if not os.path.isdir(folder_path):
            self.logger.error(f"O caminho não é um diretório: {folder_path}")
            return []

        try:
            # List all JSON files in the directory
            json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            
            if not json_files:
                self.logger.warning(f"Nenhum arquivo JSON encontrado em: {folder_path}")
                return []

            self.logger.info(f"Encontrados {len(json_files)} arquivos JSON no filesystem.")

            all_articles = []

            # Process each JSON file
            for filename in tqdm(json_files, desc="Processando arquivos JSON do filesystem"):
                try:
                    file_path = os.path.join(folder_path, filename)
                    
                    # Read and parse JSON
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Handle both list and dict formats
                    if isinstance(data, list):
                        all_articles.extend(data)
                    elif isinstance(data, dict):
                        all_articles.append(data)
                    else:
                        self.logger.warning(f"Formato inesperado em {filename}: {type(data)}")

                except Exception as e:
                    self.logger.error(f"Erro ao processar {filename}: {e}")

            if not all_articles:
                self.logger.warning("Nenhum artigo válido foi carregado.")
                return []

            self.logger.info(f"Total de artigos carregados do filesystem: {len(all_articles)}")
            self.logger.info("Processamento do filesystem concluído com sucesso!")
            return all_articles
            
        except Exception as e:
            self.logger.error(f"Erro ao acessar filesystem: {e}")
            return []

    def process_local_folder(self, folder_path):
        """
        Lê todos os arquivos JSON de uma pasta específica do MinIO.
        Args:
            folder_path: Caminho no formato "bucket_name/prefix/path" ou apenas "bucket_name"
        Returns:
            Lista de todos os artigos carregados dos arquivos JSON.
        """
        self.logger.info(f"Iniciando leitura dos arquivos do MinIO em: {folder_path}")

        # Parse bucket and prefix from folder_path
        parts = folder_path.split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        
        self.logger.info(f"Bucket: {bucket_name}, Prefix: {prefix}")

        try:
            # Initialize MinIO manager (uses PUBLIC storage by default)
            minio_manager = MinIOManager(storage_type="public")
            
            # List all objects in the bucket with the given prefix
            objects = minio_manager.list_files(bucket_name)
            
            # Filter JSON files matching the prefix
            json_files = []
            for obj in objects:
                if obj.object_name.startswith(prefix) and obj.object_name.endswith(".json"):
                    json_files.append(obj.object_name)
            
            if not json_files:
                self.logger.warning(f"Nenhum arquivo JSON encontrado no bucket '{bucket_name}' com prefix '{prefix}'")
                return []

            self.logger.info(f"Encontrados {len(json_files)} arquivos JSON no MinIO.")

            all_articles = []

            # Download and process each JSON file
            for object_name in tqdm(json_files, desc="Processando arquivos JSON do MinIO"):
                try:
                    # Create temporary file path
                    temp_file = f"/tmp/{os.path.basename(object_name)}"
                    
                    # Download file from MinIO
                    minio_manager.download_file(bucket_name, object_name, temp_file)
                    
                    # Read and parse JSON
                    with open(temp_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Handle both list and dict formats
                    if isinstance(data, list):
                        all_articles.extend(data)
                    elif isinstance(data, dict):
                        all_articles.append(data)
                    
                    # Clean up temporary file
                    os.remove(temp_file)

                except Exception as e:
                    self.logger.error(f"Erro ao processar {object_name}: {e}")

            if not all_articles:
                self.logger.warning("Nenhum artigo válido foi carregado.")
                return []

            self.logger.info(f"Total de artigos carregados: {len(all_articles)}")
            self.logger.info("Processamento do MinIO concluído com sucesso!")
            return all_articles
            
        except Exception as e:
            self.logger.error(f"Erro ao acessar MinIO: {e}")
            return []

    def collect_all_portal(self):
        """
        Coleta TODAS as notícias disponíveis de qualquer portal configurado.
        Usa o portal_name já definido em self.portal_name.
        """
        self.config = self.aux.get(self.portal_name)

        if not self.config:
            self.logger.error(f"Configuração para o portal '{self.portal_name}' não encontrada.")
            return []

        self.date_parser = DATE_PARSERS[self.config['date_parser']]
        
        # Usar max_page da configuração se disponível, senão descobrir automaticamente
        if 'max_page' in self.config and self.config['max_page']:
            self.max_page = self.config['max_page']
            self.logger.info(f"Usando max_page da configuração: {self.max_page}")
        else:
            self.max_page = self._find_max_page()
        
        # Usar min_page da configuração se disponível
        if 'min_page' in self.config and self.config['min_page']:
            self.min_page = self.config['min_page']

        self.logger.info(f"Iniciando coleta completa do portal '{self.portal_name}' ({self.max_page} páginas estimadas)...")

        all_articles_data = []
        article_urls_processed = set()

        for page_num in tqdm(range(self.min_page, self.max_page + 1), desc=f"Coletando páginas do {self.portal_name}"):
            page_url = self._get_page_url(page_num)
            soup = self._get_soup(page_url)
            if not soup:
                continue

            links = self._get_article_links(soup)
            if not links:
                self.logger.debug(f"Nenhum link encontrado na página {page_num}.")
                continue

            for url in tqdm(links, desc=f"Artigos da página {page_num}", leave=False):
                if url in article_urls_processed:
                    continue
                article_info = self._extract_article_info(url, page_num)
                if article_info:
                    all_articles_data.append(article_info)
                article_urls_processed.add(url)
                time.sleep(0.2)

        self.logger.info(f"Total de artigos coletados do {self.portal_name}: {len(all_articles_data)}")
        
        # Salvar JSONs localmente por portal
        self._save_articles_locally(all_articles_data)
        
        return all_articles_data
    
    def _save_articles_locally(self, articles):
        """
        Salva os artigos coletados em arquivos JSON locais, organizados por portal.
        
        Args:
            articles: Lista de dicionários com os artigos coletados
        """
        if not articles:
            self.logger.info("Nenhum artigo para salvar localmente.")
            return
        
        # Criar pasta para o portal
        portal_dir = self.local_download_dir / self.portal_name
        portal_dir.mkdir(exist_ok=True)
        
        # Salvar cada artigo em um arquivo JSON individual
        for idx, article in enumerate(articles, 1):
            # Gerar nome do arquivo baseado na URL ou índice
            article_id = article.get('url', '').split('/')[-1] or f"article_{idx}"
            filename = f"{article_id}.json"
            filepath = portal_dir / filename
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(article, f, ensure_ascii=False, indent=2)
            except Exception as e:
                self.logger.error(f"Erro ao salvar artigo {filename}: {e}")
        
        # Salvar também um arquivo consolidado com todos os artigos
        consolidated_file = portal_dir / f"{self.portal_name}_all_articles.json"
        try:
            with open(consolidated_file, 'w', encoding='utf-8') as f:
                json.dump(articles, f, ensure_ascii=False, indent=2)
            self.logger.info(f"✅ {len(articles)} artigos salvos em: {portal_dir}")
            self.logger.info(f"📄 Arquivo consolidado: {consolidated_file.name}")
        except Exception as e:
            self.logger.error(f"Erro ao salvar arquivo consolidado: {e}")

    def collect_all_nsc(self):
        """
        Coleta TODAS as notícias disponíveis do portal NSC.
        """
        self.portal_name = "nsc"
        self.config = self.aux.get(self.portal_name)

        if not self.config:
            self.logger.error("Configuração para o portal NSC não encontrada.")
            return []

        self.date_parser = DATE_PARSERS[self.config['date_parser']]
        
        # Usar max_page da configuração se disponível, senão descobrir automaticamente
        if 'max_page' in self.config and self.config['max_page']:
            self.max_page = self.config['max_page']
            self.logger.info(f"Usando max_page da configuração: {self.max_page}")
        else:
            self.max_page = self._find_max_page()
        
        # Usar min_page da configuração se disponível
        if 'min_page' in self.config and self.config['min_page']:
            self.min_page = self.config['min_page']

        self.logger.info(f"Iniciando coleta completa do portal NSC ({self.max_page} páginas estimadas)...")

        all_articles_data = []
        article_urls_processed = set()

        for page_num in tqdm(range(1, self.max_page), desc="Coletando páginas do NSC"):
            page_url = self._get_page_url(page_num)
            soup = self._get_soup(page_url)
            if not soup:
                continue

            links = self._get_article_links(soup)
            if not links:
                self.logger.warning(f"Nenhum link encontrado na página {page_num}.")
                continue

            for url in tqdm(links, desc=f"Artigos da página {page_num}", leave=False):
                if url in article_urls_processed:
                    continue
                article_info = self._extract_article_info(url, page_num)
                if article_info:
                    all_articles_data.append(article_info)
                article_urls_processed.add(url)
                time.sleep(0.2)

        self.logger.info(f"Total de artigos coletados do NSC: {len(all_articles_data)}")
        return all_articles_data

if __name__ == '__main__':
    import sys
    print("################## Collector Noticias Iniciado ###############", flush=True)
    sys.stdout.flush()
    processor = CollectorNoticias()
    print("Iniciando processamento de mensagens...", flush=True)
    processor.start()
