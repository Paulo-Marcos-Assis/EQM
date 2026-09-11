from datetime import datetime
from service_essentials.utils.logger import Logger  
logger = Logger(log_to_console=True)

# ==============================================================================
# FUNÇÕES AUXILIARES DE PARSING DE DATA
# ==============================================================================

def parse_iso_or_portuguese_date(date_str):
    """
    Analisa datas em formato ISO 8601 ou em português (ex: '24 jul 2025 às 14h38'
    ou '5 maio 2026'). Aceita ISO com fuso ("2025-07-24T14:38:34-03:00") e ISO
    puro ("2025-07-24"). Retorna um objeto de data.
    """
    if not date_str:
        return None
    try:
        # Formato ISO com fuso horário (ex: "2025-07-24T14:38:34-03:00")
        if 'T' in date_str:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00')).date()
        
        # ISO puro (ex: "2026-05-05")
        iso_only = date_str.split(' ')[0].strip()
        try:
            parsed = datetime.fromisoformat(iso_only)
            if parsed.year and parsed.month and parsed.day:
                return parsed.date()
        except ValueError:
            pass
        
        # Formato em português (ex: "24 jul 2025 às 14h38" ou "5 maio 2026")
        months_pt = {
            'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6,
            'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12,
            'janeiro': 1, 'fevereiro': 2, 'março': 3, 'abril': 4,
            'maio': 5, 'junho': 6, 'julho': 7, 'agosto': 8,
            'setembro': 9, 'outubro': 10, 'novembro': 11, 'dezembro': 12
        }
        date_part = date_str.split(' às ')[0]
        parts = date_part.split()
        if len(parts) == 3:
            first, second, third = parts
            # "24 jul 2025" -> dia mês ano
            month = months_pt.get(second.lower())
            if month and first.isdigit() and third.isdigit():
                return datetime(int(third), month, int(first)).date()
            # "setembro 2, 2026" -> mês dia, ano
            month = months_pt.get(first.lower())
            if month:
                day_str = second.rstrip(',')
                if day_str.isdigit() and third.isdigit():
                    return datetime(int(third), month, int(day_str)).date()
    except Exception as e:
        logger.warning(f"Falha ao analisar a data '{date_str}': {e}")
    return None

def parse_dmy_format(date_str):
    """
    Analisa datas no formato DD/MM/YYYY.
    Retorna um objeto de data.
    """
    if not date_str:
        return None
    try:
        # Formato "DD/MM/YYYY - HH:MM"
        date_only = date_str.split(' - ')[0].strip()
        return datetime.strptime(date_only, '%d/%m/%Y').date()
    except Exception as e:
        logger.warning(f"Falha ao analisar a data '{date_str}': {e}")
    return None

def parse_g1_format(date_str):
    """
    Analisa datas do G1 no formato 'DD/MM/YYYY HHhMMAtualizado...' ou similar.
    Extrai apenas a primeira data DD/MM/YYYY.
    Retorna um objeto de data.
    """
    if not date_str:
        return None
    try:
        # Extrair apenas os primeiros 10 caracteres (DD/MM/YYYY)
        date_only = date_str[:10].strip()
        return datetime.strptime(date_only, '%d/%m/%Y').date()
    except Exception as e:
        logger.warning(f"Falha ao analisar a data do G1 '{date_str}': {e}")
    return None

def parse_other_format(date_str):
    """
    Analisa datas no formato 'DD/MM/YYYY às HH:MM' ou 'DD de mês de YYYY'.
    Retorna um objeto de data.
    """
    if not date_str:
        return None
    try:
        # Formato "DD/MM/YYYY às HH:MM"
        strip_date = date_str.split("Atualizada")[0].strip()
        dt = datetime.strptime(strip_date, "%d/%m/%Y às %Hh%M")
        only_date = dt.date()
        return only_date
    except:
        pass
    
    try:
        # Formato "2 de março de 2026" (jornalconexao)
        months_pt_full = {
            'janeiro': 1, 'fevereiro': 2, 'março': 3, 'abril': 4, 
            'maio': 5, 'junho': 6, 'julho': 7, 'agosto': 8,
            'setembro': 9, 'outubro': 10, 'novembro': 11, 'dezembro': 12
        }
        
        # Remove texto extra e normaliza
        clean_date = date_str.split("Atualizada")[0].strip().lower()
        
        # Tenta extrair "DD de mês de YYYY"
        parts = clean_date.split(' de ')
        if len(parts) == 3:
            day = int(parts[0].strip())
            month_name = parts[1].strip()
            year = int(parts[2].strip())
            
            month = months_pt_full.get(month_name)
            if month:
                return datetime(year, month, day).date()
    except Exception as e:
        logger.warning(f"Falha ao analisar a data '{date_str}': {e}")
    
    return None