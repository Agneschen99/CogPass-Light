'use client';
import { useEffect, useState } from 'react';

export default function EEGStatus() {
  const [connected, setConnected] = useState(false);
  const [lastMsg, setLastMsg] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ws = new WebSocket('ws://127.0.0.1:8765');
    ws.onopen = () => { setConnected(true); setError(null); };
    ws.onmessage = (e) => {
      try { setLastMsg(JSON.parse(e.data)); }
      catch { setLastMsg(e.data); }
    };
    ws.onerror = () => setError('WebSocket error');
    ws.onclose  = () => setConnected(false);
    return () => ws.close();
  }, []);

  return (
    <div style={{padding:8, border:'1px solid #eee', borderRadius:8}}>
      <div>EEG: {connected ? 'Connected' : 'Disconnected'}{error ? ` · ${error}` : ''}</div>
      <pre style={{fontSize:12, background:'#f7f7f7', padding:8, maxHeight:200, overflow:'auto'}}>
        {lastMsg ? JSON.stringify(lastMsg, null, 2) : 'Waiting for data...'}
      </pre>
    </div>
  );
}
