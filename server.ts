import express from 'express';
import http from 'http';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { WebSocketServer, WebSocket } from 'ws';
import { GoogleGenAI, Modality } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const server = http.createServer(app);
const PORT = Number(process.env.PORT) || 3000;

app.use(express.json({ limit: '10mb' }));

// Shared Gemini client with required User-Agent
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

/**
 * Conversational Fleet Condition Endpoint
 * Analyzes real-time fleet telemetry, degradation, and work orders
 */
app.post('/api/fleet/conversation', async (req, res) => {
  try {
    const { message, fleetSummary, returnAudio = true } = req.body;

    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }

    const systemInstruction = `You are the Fleet Chief Diagnostic AI for a connected fleet of heavy haul trucks, city transit buses, and industrial compressors.
You continuously monitor 5 core streaming sensor channels:
1. Vibration (mm/s RMS) - Mechanical unbalance, bearing race pitting
2. Temperature (°C) - Bearing and thermodynamic distress
3. Motor Current (Amps) - Mechanical resistance surge
4. Discharge Pressure (PSI) - Seal/valve clearance wear
5. RPM - Rotational speed slippage

CURRENT FLEET TELEMETRY CONTEXT:
${JSON.stringify(fleetSummary, null, 2)}

INSTRUCTIONS:
- Answer the user's inquiry directly, accurately, and concisely.
- Cite specific machines (e.g. TRK-101, BUS-204, CMP-302, CMP-309), their health score, remaining useful life (RUL in cycles), and specific sensor values (vibration, temp, current, pressure, rpm).
- Explicitly mention any active or pending work orders, especially critical exceptions requiring supervisor sign-off.
- Keep the tone professional, authoritative, and operational (like a senior fleet maintenance lead). Limit spoken responses to 2-4 focused sentences unless detailed breakdown is requested.`;

    // Generate conversational diagnostic text
    const textResponse = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: message,
      config: {
        systemInstruction,
        temperature: 0.4,
      },
    });

    const replyText = textResponse.text || 'Fleet telemetry is currently operating nominal.';

    let audioBase64: string | null = null;
    if (returnAudio) {
      try {
        // Generate natural spoken response via Gemini TTS
        const ttsResponse = await ai.models.generateContent({
          model: 'gemini-3.8-flash-lite-tts',
          contents: [
            {
              role: 'user',
              parts: [
                {
                  text: replyText.replace(/[*_#`]/g, ''), // clean markdown for speech
                  speechMetadata: {
                    style: 'Clear, authoritative fleet diagnostic lead',
                  },
                },
              ],
            },
          ],
          config: {
            responseModalities: ['AUDIO'],
            speechConfig: {
              voiceConfig: {
                prebuiltVoiceConfig: { voiceName: 'Zephyr' },
              },
            },
          },
        });

        audioBase64 = ttsResponse.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data || null;
      } catch (audioErr) {
        console.warn('TTS generation notice:', audioErr);
      }
    }

    return res.json({
      text: replyText,
      audio: audioBase64,
    });
  } catch (error: any) {
    console.error('Fleet conversation error:', error);
    return res.status(500).json({
      error: error.message || 'Failed to analyze fleet condition',
    });
  }
});

/**
 * Text-to-Speech endpoint for generated text
 */
app.post('/api/fleet/tts', async (req, res) => {
  try {
    const { text, voice = 'Zephyr' } = req.body;
    if (!text) {
      return res.status(400).json({ error: 'Text is required' });
    }

    const ttsResponse = await ai.models.generateContent({
      model: 'gemini-3.8-flash-lite-tts',
      contents: [
        {
          role: 'user',
          parts: [{ text: text.replace(/[*_#`]/g, '') }],
        },
      ],
      config: {
        responseModalities: ['AUDIO'],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName: voice },
          },
        },
      },
    });

    const audioBase64 = ttsResponse.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data || null;
    return res.json({ audio: audioBase64 });
  } catch (error: any) {
    console.error('TTS error:', error);
    return res.status(500).json({ error: error.message || 'TTS generation failed' });
  }
});

// Setup WebSocket server for Gemini Live API (gemini-3.8-live) real-time voice streaming
const wss = new WebSocketServer({ noServer: true });

server.on('upgrade', (request, socket, head) => {
  const pathname = request.url ? new URL(request.url, `http://${request.headers.host}`).pathname : '';
  if (pathname === '/api/live' || pathname === '/live') {
    wss.handleUpgrade(request, socket, head, (ws) => {
      wss.emit('connection', ws, request);
    });
  }
});

wss.on('connection', async (clientWs: WebSocket) => {
  let liveSession: any = null;

  try {
    liveSession = await ai.live.connect({
      model: 'gemini-3.8-live',
      config: {
        responseModalities: [Modality.AUDIO],
        speechConfig: {
          voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } },
        },
        systemInstruction: `You are the voice assistant for the Connected Fleet Predictive Maintenance platform.
You analyze connected heavy haul trucks, transit buses, and industrial compressors streaming vibration, temperature, motor current, pressure, and RPM.
Answer questions about machine health, remaining useful life, and work orders in a concise, conversational manner.`,
      },
      callbacks: {
        onmessage: (message: any) => {
          const audio = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
          const textPart = message.serverContent?.modelTurn?.parts?.find((p: any) => p.text);
          if (audio && clientWs.readyState === WebSocket.OPEN) {
            clientWs.send(JSON.stringify({ audio, text: textPart?.text }));
          }
          if (message.serverContent?.interrupted && clientWs.readyState === WebSocket.OPEN) {
            clientWs.send(JSON.stringify({ interrupted: true }));
          }
        },
      },
    });

    clientWs.on('message', (data: Buffer | string) => {
      try {
        const parsed = JSON.parse(data.toString());
        if (parsed.audio && liveSession) {
          liveSession.sendRealtimeInput({
            audio: { data: parsed.audio, mimeType: 'audio/pcm;rate=16000' },
          });
        }
        if (parsed.text && liveSession) {
          liveSession.sendRealtimeInput({
            text: parsed.text,
          });
        }
      } catch (err) {
        console.error('Error forwarding message to Live API:', err);
      }
    });

    clientWs.on('close', () => {
      if (liveSession) {
        try {
          liveSession.close();
        } catch (e) {
          // session closed
        }
      }
    });
  } catch (err: any) {
    console.error('Failed to connect to gemini-3.8-live:', err);
    if (clientWs.readyState === WebSocket.OPEN) {
      clientWs.send(JSON.stringify({ error: err.message || 'Live session initialization failed' }));
      clientWs.close();
    }
  }
});

// Mount Vite or static serving
async function setupServer() {
  const isDev = process.env.NODE_ENV !== 'production';

  if (isDev) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true, hmr: process.env.DISABLE_HMR !== 'true' },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  server.listen(PORT, '0.0.0.0', () => {
    console.log(`Predictive Maintenance Fleet Server listening on http://0.0.0.0:${PORT}`);
  });
}

setupServer();
