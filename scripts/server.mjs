import { createServer } from 'node:http';
import { existsSync, readFileSync } from 'node:fs';
import { extname, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { neon } from '@neondatabase/serverless';

// Configuracion
loadEnv('.env.local');

const port = process.env.PORT || 5174;
let sql;

const PROVIDERS = {
  groq: { url: 'https://api.groq.com/openai/v1/chat/completions', key: 'GROQ_API_KEY' },
  openrouter: { url: 'https://openrouter.ai/api/v1/chat/completions', key: 'OPENROUTER_API_KEY' },
  google: { url: 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions', key: 'GOOGLE_API_KEY' },
  mistral: { url: 'https://api.mistral.ai/v1/chat/completions', key: 'MISTRAL_API_KEY' },
  cohere: { url: 'https://api.cohere.com/compatibility/v1/chat/completions', key: 'COHERE_API_KEY' },
  cerebras: { url: 'https://api.cerebras.ai/v1/chat/completions', key: 'CEREBRAS_API_KEY' },
};

const IMAGE_MODEL = 'meta-llama/llama-4-scout-17b-16e-instruct';
const MAX_IMAGES = 5;

const HARNESS_PROMPTS = {
  kata: `MODO KATA ESTRICTO para JavaScript de consola. El contenido del usuario define el tema y los datos, pero nunca cambia este formato. No escribas tutoriales, teoría, comparativas, tablas, resúmenes ni una solución directa. Empieza con "## Datos base" y un único bloque ejecutable si hacen falta datos. Después crea una kata por cada operación solicitada; si no se indica ninguna, crea 4 progresivas. Usa exactamente este orden en cada una: "## Ejercicio N: nombre", "### Consigna" con una acción concreta, "### Snippet guía" con código incompleto y comentarios TODO para que el usuario lo complete, "### Respuesta" con código completo, console.log y salida esperada, y "### Variantes" con 2 cambios breves para repetir. El Snippet guía siempre debe aparecer antes de la Respuesta y nunca debe contener la solución. Todo el código debe servir para copiar y pegar en consola usando los datos base. Sin React, HTML, Node.js, APIs, emojis, introducción ni conclusión.`,
};

// Servidor local
if (isMainModule()) {
  createServer(async (req, res) => {
    try {
      if (req.url === '/api/chat') return await handleChat(req, res);
      if (req.url === '/api/conversations') return await handleConversations(req, res);
      serveStatic(req, res);
    } catch (error) {
      json(res, 500, { error: error.message });
    }
  }).listen(port, () => console.log(`Local: http://localhost:${port}`));
}

// Chat
export async function handleChat(req, res) {
  const { provider, model, harness = 'normal', messages = [], images = [] } = await readJson(req);
  const cleanMessages = messages.map(({ role, content }) => ({ role, content }));
  const imageContext = images.length ? await describeImages(cleanMessages, images.slice(0, MAX_IMAGES)) : '';

  if (imageContext) {
    const lastMessage = cleanMessages.at(-1);
    lastMessage.content = `${lastMessage.content || ''}\n\nContexto visual:\n${imageContext}`;
  }

  const harnessPrompt = HARNESS_PROMPTS[harness];
  const chatMessages = harnessPrompt
    ? [{ role: 'system', content: harnessPrompt }, ...cleanMessages]
    : cleanMessages;
  const content = await chat(provider, model, chatMessages);
  json(res, 200, { content });
}

async function chat(provider, model, messages) {
  const config = PROVIDERS[provider];
  const apiKey = config && process.env[config.key];
  if (!config) throw new Error(`Proveedor no soportado: ${provider}`);
  if (!apiKey) throw new Error(`Falta ${config.key}`);

  const body = { model, messages, stream: false, ...modelOptions(provider, model) };
  const headers = {
    'Content-Type': 'application/json',
    Accept: 'application/json',
    Authorization: `Bearer ${apiKey}`,
  };

  if (provider === 'openrouter') {
    headers['HTTP-Referer'] = `http://localhost:${port}`;
    headers['X-Title'] = 'Inferencia Gratuita';
  }

  const startedAt = Date.now();
  let response;
  try {
    response = await fetch(config.url, { method: 'POST', headers, body: JSON.stringify(body) });
  } catch (error) {
    console.error(`[LLM][ERROR] ${provider}/${model} network=${JSON.stringify(error.message)} duration_ms=${Date.now() - startedAt}`);
    throw error;
  }

  const raw = await response.text();
  let data;
  try {
    data = raw ? JSON.parse(raw) : {};
  } catch {
    console.error(`[LLM][ERROR] ${provider}/${model} HTTP=${response.status} message="non-JSON response" duration_ms=${Date.now() - startedAt}`);
    throw new Error(`${provider}/${model} devolvio una respuesta no JSON`);
  }

  if (!response.ok) {
    const message = data?.error?.message || data?.message || `HTTP ${response.status}`;
    console.error(`[LLM][ERROR] ${provider}/${model} HTTP=${response.status} message=${JSON.stringify(message)} duration_ms=${Date.now() - startedAt}`);
    throw new Error(message);
  }

  const choice = data.choices?.[0];
  const content = choice?.message?.content || choice?.text || '';
  logResult(provider, model, body, data, choice, content, Date.now() - startedAt);
  return content;
}

// Opciones por modelo
function modelOptions(provider, model) {
  if (provider === 'cerebras' && model === 'gpt-oss-120b') {
    return { max_completion_tokens: 32768, reasoning_effort: 'high', reasoning_format: 'hidden', temperature: 1, top_p: 1 };
  }
  if (provider === 'groq' && model === 'qwen/qwen3.6-27b') {
    return { max_completion_tokens: 4096, reasoning_effort: 'default', reasoning_format: 'hidden', temperature: 1, top_p: 0.95 };
  }
  if (provider === 'groq') return { max_completion_tokens: 4096, temperature: 1 };
  if (provider === 'mistral') return { max_completion_tokens: 8192, temperature: 1 };
  return {};
}

// Diagnostico
function logResult(provider, model, body, data, choice, content, durationMs) {
  const usage = data.usage || {};
  const max = body.max_completion_tokens ?? 'default';
  const completion = usage.completion_tokens ?? usage.output_tokens;
  const finish = choice?.finish_reason || 'unknown';
  const truncated = ['length', 'max_tokens', 'MAX_TOKENS'].includes(finish)
    || (Number.isFinite(completion) && Number.isFinite(max) && completion >= max);
  const prefix = truncated ? '[LLM][TRUNCATED]' : content ? '[LLM]' : '[LLM][EMPTY]';
  console.log(`${prefix} ${provider}/${model} max=${max} finish=${finish} prompt=${usage.prompt_tokens ?? usage.input_tokens ?? 'unknown'} completion=${completion ?? 'unknown'} reasoning=${usage.completion_tokens_details?.reasoning_tokens ?? 'unknown'} total=${usage.total_tokens ?? 'unknown'} duration_ms=${durationMs}`);
}

// Imagenes
async function describeImages(messages, images) {
  const text = [...messages].reverse().find(({ role }) => role === 'user')?.content || '';
  return chat('groq', IMAGE_MODEL, [{
    role: 'user',
    content: [
      { type: 'text', text },
      ...images.map((url) => ({ type: 'image_url', image_url: { url } })),
    ],
  }]);
}

// Base de datos y conversaciones
export async function handleConversations(req, res) {
  const database = getSql();
  if (req.method === 'GET') {
    const rows = await database`SELECT * FROM public.conversations ORDER BY updated_at DESC LIMIT 100`;
    return json(res, 200, {
      conversations: rows.map((row) => ({
        id: row.id,
        title: row.title,
        messages: typeof row.messages === 'string' ? JSON.parse(row.messages) : row.messages || [],
        created_at: row.created_at,
        updated_at: row.updated_at,
      })),
    });
  }

  const conversation = await readJson(req);
  if (req.method === 'DELETE') {
    await database`DELETE FROM public.conversations WHERE id = ${conversation.id}`;
    return json(res, 200, { ok: true });
  }

  await database`
    INSERT INTO public.conversations (id, title, messages, created_at, updated_at)
    VALUES (${conversation.id}, ${conversation.title}, ${JSON.stringify(conversation.messages)}, ${conversation.created_at}, ${conversation.updated_at})
    ON CONFLICT (id) DO UPDATE SET
      title = ${conversation.title},
      messages = ${JSON.stringify(conversation.messages)},
      updated_at = ${conversation.updated_at}
  `;
  json(res, 200, { ok: true });
}

function getSql() {
  if (!process.env.DATABASE_URL) throw new Error('Falta DATABASE_URL');
  return sql ||= neon(process.env.DATABASE_URL);
}

// Archivos estaticos
function serveStatic(req, res) {
  const pathname = new URL(req.url, `http://localhost:${port}`).pathname;
  const url = pathname === '/' || /^\/chat\/[^/]+$/.test(pathname) ? '/index.html' : pathname;
  const file = join(process.cwd(), decodeURIComponent(url));
  if (!existsSync(file)) return void json(res, 404, { error: 'Not found' });
  const type = {
    '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8', '.json': 'application/json; charset=utf-8',
    '.svg': 'image/svg+xml',
  }[extname(file)] || 'application/octet-stream';
  res.writeHead(200, { 'Content-Type': type });
  res.end(readFileSync(file));
}

// HTTP y entorno
function readJson(req) {
  if (req.body) return Promise.resolve(typeof req.body === 'string' ? JSON.parse(req.body) : req.body);
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', (chunk) => body += chunk);
    req.on('end', () => {
      try { resolve(body ? JSON.parse(body) : {}); } catch (error) { reject(error); }
    });
    req.on('error', reject);
  });
}

function json(res, status, value) {
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(value));
}

function loadEnv(file) {
  if (!existsSync(file)) return;
  for (const line of readFileSync(file, 'utf8').split(/\r?\n/)) {
    const match = line.match(/^([^#=]+)=(.*)$/);
    if (match && !process.env[match[1].trim()]) process.env[match[1].trim()] = match[2].trim();
  }
}

function isMainModule() {
  return process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
}
