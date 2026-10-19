# Ingestão e busca semântica com LangChain e PostgreSQL

Aplicação de perguntas e respostas sobre um PDF. A ingestão extrai o texto, divide em trechos de 1.000 caracteres com sobreposição de 150, gera embeddings e os armazena no PostgreSQL com pgvector. O chat recupera até 10 trechos relevantes e envia o contexto à LLM configurada.

## Requisitos

- Windows 10/11, PowerShell e Python 3.10 ou superior (recomendado Python 3.12).
- Docker Desktop instalado e em execução, com Docker Compose.
- Uma chave de API do Google Gemini ou da OpenAI.

### Instalar Python no Windows

Baixe o instalador oficial em [python.org/downloads/windows](https://www.python.org/downloads/windows/). Durante a instalação, marque **Add python.exe to PATH**. Feche e abra novamente o PowerShell e confirme:

```powershell
python --version
```

O comando deve mostrar Python 3.10 ou superior. O projeto também funciona em macOS/Linux, com os comandos equivalentes indicados nas seções abaixo.

## Criar uma chave de API

Escolha um provedor e crie a chave na página oficial:

- **Google Gemini:** [Google AI Studio — criar/gerenciar API keys](https://aistudio.google.com/app/apikey). Consulte também o [guia oficial de chaves Gemini](https://ai.google.dev/gemini-api/docs/api-key).
- **OpenAI:** [OpenAI Platform — API keys](https://platform.openai.com/api-keys). O [guia do Help Center](https://help.openai.com/en/articles/4936850-how-to-set-up-your-chatgpt-account) explica como criar e guardar a chave.

Neste projeto, “token” significa **chave de API**. Uma senha, cookie ou token de login do ChatGPT não serve. A chave da API OpenAI é criada na plataforma da API; a assinatura do ChatGPT e o uso da API têm faturamento separado. Veja [como funciona a cobrança de ChatGPT e API](https://help.openai.com/en/articles/9039756-managing-billing-for-chatgpt-and-the-api-platform).

Guarde a chave em local seguro e não a compartilhe nem a envie ao GitHub. O arquivo `.env` é ignorado pelo Git.

## Instalação e configuração (Windows PowerShell)

Abra o PowerShell na pasta do projeto. Crie e ative um ambiente virtual:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Se o PowerShell bloquear a ativação, permita scripts locais somente nesta janela e ative novamente:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

Instale as dependências e crie seu arquivo local de configuração:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Abra `.env` em um editor e configure **um** provedor:

**Para Gemini (padrão):**

```dotenv
LLM_PROVIDER=gemini
GOOGLE_API_KEY=cole_a_chave_do_google_aqui
```

**Ou para OpenAI:**

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=cole_a_chave_da_openai_aqui
```

Os nomes dos modelos, a collection e a conexão com o banco já têm valores padrão em `.env.example`. Use o mesmo provedor e modelo de embeddings na ingestão e na busca. Se trocar o modelo de embeddings depois da ingestão, apague a collection/volume e ingira novamente, pois a dimensão dos vetores pode mudar.

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Edite `.env` e preencha a chave do provedor escolhido, da mesma forma mostrada acima.

## Executar o projeto

Mantenha o Docker Desktop aberto. Com o ambiente virtual ativado e `.env` configurado, inicie o PostgreSQL com pgvector:

```powershell
docker compose up -d
```

Confirme que o serviço está saudável:

```powershell
docker compose ps
```

O serviço `postgres` deve aparecer como `healthy`. O serviço `bootstrap_vector_ext` cria a extensão `vector` e encerra com código `0`; esse encerramento é esperado.

Ingere o PDF padrão (`document.pdf`):

```powershell
python src/ingest.py
```

Quando a ingestão terminar, abra o chat:

```powershell
python src/chat.py
```

Digite uma pergunta sobre algo que aparece no PDF e pressione Enter. Digite `sair` para encerrar. Se perguntar sobre algo que não consta no documento, o chat deve informar que não encontrou dados suficientes.

Para parar o banco quando terminar:

```powershell
docker compose down
```

Esse comando mantém os dados no volume Docker. Para remover também os dados armazenados no banco, use `docker compose down -v`; isso apaga a collection e você precisará ingerir o PDF novamente.

## Configurações principais

As variáveis ficam no arquivo `.env`:

| Variável | Uso |
| --- | --- |
| `LLM_PROVIDER` | `gemini` ou `openai` |
| `GOOGLE_API_KEY` | Chave de API do Gemini |
| `OPENAI_API_KEY` | Chave de API OpenAI |
| `DATABASE_URL` | Endereço do PostgreSQL local iniciado pelo Compose |
| `PG_VECTOR_COLLECTION_NAME` | Nome da collection vetorial |
| `PDF_PATH` | Caminho do PDF a ingerir; padrão `document.pdf` |
| `INGEST_BATCH_SIZE` | Chunks enviados por lote durante a ingestão |
| `INGEST_BATCH_DELAY_SECONDS` | Pausa entre lotes |

## Estrutura

- `document.pdf`: arquivo de entrada padrão.
- `src/ingest.py`: carrega o PDF, divide o texto e armazena os vetores.
- `src/search.py`: configura embeddings/LLM, consulta pgvector e monta o prompt.
- `src/chat.py`: interação pelo terminal.
- `docker-compose.yml`: PostgreSQL 17 com pgvector.
- `.env.example`: modelo de configuração sem chaves.
