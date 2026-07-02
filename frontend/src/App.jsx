import React, { useState, useEffect, useRef } from 'react';
import { 
  MessageSquare, Folder, Zap, Cpu, Box, Database, 
  Activity, BarChart2, Terminal, Settings, Play, 
  Trash2, Download, AlertCircle, RefreshCw, Layers
} from 'lucide-react';

const API_BASE = 'http://localhost:8000';
const WS_BASE = 'ws://localhost:8000';

// Helper component for SVG Sparkline Chart
const Sparkline = ({ data, color, maxVal, unit }) => {
  if (!data || data.length < 2) {
    return (
      <div style={{ height: 80, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b', fontSize: 12 }}>
        Waiting for data...
      </div>
    );
  }
  const width = 300;
  const height = 80;
  const max = maxVal || Math.max(...data, 1);
  const min = 0;
  const points = data.map((val, index) => {
    const x = (index / (data.length - 1)) * width;
    const y = height - ((val - min) / (max - min)) * height * 0.7 - height * 0.15;
    return `${x},${y}`;
  }).join(' ');

  return (
    <div style={{ position: 'relative', width: '100%' }}>
      <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} style={{ overflow: 'visible' }}>
        {/* Glow */}
        <polyline fill="none" stroke={color} strokeWidth="4" strokeOpacity="0.25" points={points} />
        {/* Main Line */}
        <polyline fill="none" stroke={color} strokeWidth="2" points={points} />
        {/* Endpoint Dot */}
        {data.length > 0 && (
          <circle 
            cx={width} 
            cy={height - ((data[data.length - 1] - min) / (max - min)) * height * 0.7 - height * 0.15} 
            r="4" 
            fill={color} 
            style={{ filter: `drop-shadow(0 0 4px ${color})` }}
          />
        )}
      </svg>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#64748b', marginTop: 4 }}>
        <span>0</span>
        <span>Current: {data[data.length - 1]?.toFixed(1)} {unit}</span>
        <span>Peak: {max.toFixed(1)} {unit}</span>
      </div>
    </div>
  );
};

export default function App() {
  const [activePage, setActivePage] = useState('chat');
  const [activeModel, setActiveModel] = useState(null);
  const [models, setModels] = useState([]);
  const [downloadModelId, setDownloadModelId] = useState('');
  
  // Settings States
  const [engineSettings, setEngineSettings] = useState({
    runtime: { max_new_tokens: 128, temperature: 0.7, top_p: 0.95 },
    cache: { gpu_limit: 'auto', ram_limit: 'auto', expert_limit: 'auto' },
    memory: { max_vram_mb: 5800, max_ram_percent: 35 },
    execution: { dtype: 'fp8', profiling: true }
  });

  // Chat States
  const [messages, setMessages] = useState([
    { role: 'assistant', content: "Hello! I am Turbo-LLM. Ask me anything, or load a model to begin inference!" }
  ]);
  const [inputText, setInputText] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('You are a helpful, respectful, and honest assistant.');
  const [isGenerating, setIsGenerating] = useState(false);
  const [thinkingParam, setThinkingParam] = useState('off');
  const [streamingParam, setStreamingParam] = useState(true);
  const [thinkingOutput, setThinkingOutput] = useState('');
  const [generationOutput, setGenerationOutput] = useState('');

  // Live Metrics & Websockets States
  const [systemStats, setSystemStats] = useState({
    status: 'Unloaded',
    active_model: null,
    ram_usage_percent: 0,
    ram_used_mb: 0,
    ram_total_mb: 0,
    cpu_percent: 0,
    vram_allocated_mb: 0,
    vram_total_mb: 0,
    vram_free_mb: 0,
    gpu_hits: 0,
    ram_hits: 0,
    ssd_hits: 0,
    cache_limit: 0,
    cache_count: 0
  });

  const [tpsHistory, setTpsHistory] = useState([]);
  const [vramHistory, setVramHistory] = useState([]);
  const [ramHistory, setRamHistory] = useState([]);
  const [cpuHistory, setCpuHistory] = useState([]);
  const [ssdReadsHistory, setSsdReadsHistory] = useState([]);
  const [logsList, setLogsList] = useState([]);
  const [selectedLogFilter, setSelectedLogFilter] = useState('all');
  const [pipelineLayers, setPipelineLayers] = useState({});
  const [expertStats, setExpertStats] = useState([]);
  
  // Selected Layer modal for Layer Timeline
  const [selectedLayerDetail, setSelectedLayerDetail] = useState(null);

  // Benchmarks States
  const [benchmarkPrompt, setBenchmarkPrompt] = useState('Explain what a mixture-of-experts transformer is.');
  const [benchmarkTokens, setBenchmarkTokens] = useState(64);
  const [pastBenchmarks, setPastBenchmarks] = useState([]);
  const [comparedBenchmarks, setComparedBenchmarks] = useState([]);
  const [isBenchmarking, setIsBenchmarking] = useState(false);

  const logsEndRef = useRef(null);
  const chatEndRef = useRef(null);
  const activeReader = useRef(null);

  const filterControlTokens = (text) => {
    if (!text) return "";
    return text
      .replace(/<\|im_end\|>/g, "")
      .replace(/<\|im_start\|>/g, "")
      .replace(/<\|endoftext\|>/g, "")
      .replace(/im-end/g, "")
      .replace(/im_end/g, "");
  };

  // Load models on startup
  useEffect(() => {
    fetchModels();
    fetchSettings();
    fetchBenchmarks();
    setupWebsockets();
  }, []);

  // Scroll logs to bottom
  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logsList]);

  // Scroll chat to bottom
  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, generationOutput, thinkingOutput]);

  const fetchModels = async () => {
    try {
      const res = await fetch(`${API_BASE}/v1/models`);
      const data = await res.json();
      setModels(data.data || []);
      const running = data.data.find(m => m.status === 'Running');
      if (running) {
        setActiveModel(running.id);
      } else {
        setActiveModel(null);
      }
    } catch (e) {
      console.error("Error fetching models:", e);
    }
  };

  const fetchSettings = async () => {
    try {
      const res = await fetch(`${API_BASE}/v1/settings`);
      const data = await res.json();
      if (data.settings) {
        setEngineSettings(data.settings);
      }
    } catch (e) {
      console.error("Error fetching settings:", e);
    }
  };

  const fetchBenchmarks = async () => {
    try {
      const res = await fetch(`${API_BASE}/v1/benchmark`);
      const data = await res.json();
      setPastBenchmarks(data.benchmarks || []);
    } catch (e) {
      console.error("Error fetching benchmarks:", e);
    }
  };

  const setupWebsockets = () => {
    // 1. Stats WebSocket
    const statsWs = new WebSocket(`${WS_BASE}/ws/stats`);
    statsWs.onmessage = (event) => {
      const stats = JSON.parse(event.data);
      setSystemStats(stats);
      
      // Update charts histories (cap at 40 points)
      setVramHistory(prev => [...prev.slice(-39), stats.vram_allocated_mb]);
      setRamHistory(prev => [...prev.slice(-39), stats.ram_used_mb]);
      setCpuHistory(prev => [...prev.slice(-39), stats.cpu_percent]);
      setSsdReadsHistory(prev => [...prev.slice(-39), stats.ssd_hits]);

      if (stats.last_metrics && stats.last_metrics.tps) {
        setTpsHistory(prev => [...prev.slice(-39), stats.last_metrics.tps]);
      }

      if (stats.status === 'Loaded' && activeModel === null && stats.active_model) {
        setActiveModel(stats.active_model);
      } else if (stats.status === 'Unloaded') {
        setActiveModel(null);
      }
    };

    // 2. Logs WebSocket
    const logsWs = new WebSocket(`${WS_BASE}/ws/logs`);
    logsWs.onmessage = (event) => {
      const log = JSON.parse(event.data);
      setLogsList(prev => [...prev.slice(-199), log.message]);
    };

    // 3. Pipeline / Layer progress WebSocket
    const pipeWs = new WebSocket(`${WS_BASE}/ws/pipeline`);
    pipeWs.onmessage = (event) => {
      const update = JSON.parse(event.data);
      if (update.type === 'layer_progress') {
        setPipelineLayers(prev => ({
          ...prev,
          [update.layer_id]: update
        }));
      }
    };

    // 4. Experts WebSocket
    const expWs = new WebSocket(`${WS_BASE}/ws/experts`);
    expWs.onmessage = (event) => {
      const update = JSON.parse(event.data);
      if (update.type === 'experts') {
        setExpertStats(update.experts || []);
      }
    };

    return () => {
      statsWs.close();
      logsWs.close();
      pipeWs.close();
      expWs.close();
    };
  };

  const handleLoadModel = async (modelId) => {
    try {
      setSystemStats(prev => ({ ...prev, status: 'Loading' }));
      const res = await fetch(`${API_BASE}/v1/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelId, config: engineSettings })
      });
      if (res.ok) {
        setTimeout(fetchModels, 3000);
      }
    } catch (e) {
      console.error("Error loading model:", e);
    }
  };

  const handleUnloadModel = async () => {
    try {
      const res = await fetch(`${API_BASE}/v1/unload`, { method: 'POST' });
      if (res.ok) {
        setActiveModel(null);
        fetchModels();
      }
    } catch (e) {
      console.error("Error unloading model:", e);
    }
  };

  const handleDownloadModel = async () => {
    if (!downloadModelId) return;
    try {
      const res = await fetch(`${API_BASE}/v1/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: downloadModelId })
      });
      if (res.ok) {
        alert("Download started in the background. Check logs page for progress.");
        setDownloadModelId('');
        fetchModels();
      }
    } catch (e) {
      console.error("Error downloading model:", e);
    }
  };

  const handleSaveSettings = async (newSettings) => {
    try {
      const res = await fetch(`${API_BASE}/v1/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ settings: newSettings })
      });
      if (res.ok) {
        setEngineSettings(newSettings);
      }
    } catch (e) {
      console.error("Error saving settings:", e);
    }
  };

  const handleCancelGeneration = () => {
    if (activeReader.current) {
      try {
        activeReader.current.cancel();
      } catch (err) {
        console.error("Error canceling reader:", err);
      }
      activeReader.current = null;
      setIsGenerating(false);
      
      const cleanContent = filterControlTokens(generationOutput);
      const cleanThink = filterControlTokens(thinkingOutput);
      
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: cleanContent + " [Generation Canceled]", 
        thinkingContent: cleanThink 
      }]);
      setGenerationOutput('');
      setThinkingOutput('');
    }
  };

  const handleSendMessage = async () => {
    if (!inputText.trim() || isGenerating) return;
    
    const userMsg = inputText;
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setInputText('');
    setIsGenerating(true);
    setThinkingOutput('');
    setGenerationOutput('');

    // Reset pipeline layers visualizer for a fresh token timeline
    setPipelineLayers({});

    try {
      const payloadMessages = [
        ...(systemPrompt ? [{ role: 'system', content: systemPrompt }] : []),
        ...messages.filter(m => m.role !== 'system'),
        { role: 'user', content: userMsg }
      ];

      const res = await fetch(`${API_BASE}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: activeModel || "turbo-llm",
          messages: payloadMessages.map(m => ({ role: m.role, content: m.content })),
          temperature: engineSettings.runtime.temperature,
          top_p: engineSettings.runtime.top_p,
          max_tokens: engineSettings.runtime.max_new_tokens,
          stream: streamingParam,
          thinking: thinkingParam
        })
      });

      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || "Inference failed");
      }

      if (streamingParam) {
        const reader = res.body.getReader();
        activeReader.current = reader;
        const decoder = new TextDecoder();
        let buffer = '';
        let accumulatedText = '';
        let accumulatedThinking = '';
        let isThinkingMode = false;

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.startsWith('data: [DONE]')) {
              break;
            }
            if (line.startsWith('data: ')) {
              try {
                const chunk = JSON.parse(line.slice(6));
                const delta = chunk.choices[0].delta.content || '';
                
                let tempDelta = delta;
                
                if (tempDelta.includes('<think>')) {
                  isThinkingMode = true;
                  const parts = tempDelta.split('<think>');
                  accumulatedText += parts[0];
                  tempDelta = parts[1] || '';
                }
                
                if (isThinkingMode) {
                  if (tempDelta.includes('</think>')) {
                    isThinkingMode = false;
                    const parts = tempDelta.split('</think>');
                    accumulatedThinking += parts[0];
                    tempDelta = parts[1] || '';
                    accumulatedText += tempDelta;
                  } else {
                    accumulatedThinking += tempDelta;
                  }
                } else {
                  accumulatedText += tempDelta;
                }

                setGenerationOutput(filterControlTokens(accumulatedText));
                setThinkingOutput(filterControlTokens(accumulatedThinking));
              } catch (e) {
                // Ignore parsing issues for empty tokens
              }
            }
          }
        }
        
        // Finalize streaming
        const cleanContent = filterControlTokens(accumulatedText);
        const cleanThink = filterControlTokens(accumulatedThinking);
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: cleanContent, 
          thinkingContent: cleanThink 
        }]);
        setGenerationOutput('');
        setThinkingOutput('');
      } else {
        const data = await res.json();
        setMessages(prev => [...prev, { role: 'assistant', content: filterControlTokens(data.choices[0].message.content) }]);
      }
    } catch (e) {
      console.error(e);
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${e.message}. Make sure a model is loaded in the Models tab.` }]);
    } finally {
      setIsGenerating(false);
      activeReader.current = null;
      fetchModels(); // Refresh stats
    }
  };

  const runBenchmark = async () => {
    if (isBenchmarking || !activeModel) return;
    setIsBenchmarking(true);
    try {
      const res = await fetch(`${API_BASE}/v1/benchmark`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: activeModel,
          prompt: benchmarkPrompt,
          max_tokens: benchmarkTokens,
          runs: 1,
          warmup: true,
          context: 1024,
          temperature: 0.0
        })
      });
      if (res.ok) {
        fetchBenchmarks();
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsBenchmarking(false);
    }
  };

  const handleBenchmarkCompareToggle = (id) => {
    setComparedBenchmarks(prev => {
      if (prev.includes(id)) {
        return prev.filter(x => x !== id);
      }
      return [...prev, id];
    });
  };

  // Helper to colorize and format raw logs
  const getLogClass = (line) => {
    const lower = line.toLowerCase();
    if (lower.includes('error') || lower.includes('fail')) return 'log-line error';
    if (lower.includes('warning') || lower.includes('oom')) return 'log-line warn';
    if (lower.includes('step') || lower.includes('token:')) return 'log-line debug';
    if (lower.includes('gpu') || lower.includes('vram')) return 'log-line gpu';
    if (lower.includes('ssd') || lower.includes('ram cache') || lower.includes('hit')) return 'log-line storage';
    return 'log-line info';
  };

  const filteredLogs = logsList.filter(log => {
    if (selectedLogFilter === 'all') return true;
    const lower = log.toLowerCase();
    if (selectedLogFilter === 'engine') return lower.includes('step') || lower.includes('token') || lower.includes('generation');
    if (selectedLogFilter === 'storage') return lower.includes('ssd') || lower.includes('ram') || lower.includes('cache') || lower.includes('evict');
    if (selectedLogFilter === 'gpu') return lower.includes('cuda') || lower.includes('vram') || lower.includes('alloc') || lower.includes('slot');
    if (selectedLogFilter === 'errors') return lower.includes('error') || lower.includes('fail') || lower.includes('exception');
    return true;
  });

  return (
    <div className="app-container">
      {/* SIDEBAR */}
      <div className="sidebar">
        <div className="sidebar-brand">
          <div className="sidebar-logo">
            <Cpu size={18} color="#04060d" strokeWidth={2.5} />
          </div>
          <span className="sidebar-logo-text">Turbo-LLM</span>
        </div>

        <ul className="sidebar-menu">
          <li className={`sidebar-item ${activePage === 'chat' ? 'active' : ''}`} onClick={() => setActivePage('chat')}>
            <MessageSquare size={16} /> 💬 Chat
          </li>
          <li className={`sidebar-item ${activePage === 'models' ? 'active' : ''}`} onClick={() => setActivePage('models')}>
            <Folder size={16} /> 📁 Models
          </li>
          <li className={`sidebar-item ${activePage === 'engine' ? 'active' : ''}`} onClick={() => setActivePage('engine')}>
            <Zap size={16} /> ⚡ Engine Settings
          </li>
          <li className={`sidebar-item ${activePage === 'kv' ? 'active' : ''}`} onClick={() => setActivePage('kv')}>
            <Database size={16} /> 🧠 KV Cache
          </li>
          <li className={`sidebar-item ${activePage === 'moe' ? 'active' : ''}`} onClick={() => setActivePage('moe')}>
            <Box size={16} /> 📦 MoE Page
          </li>
          <li className={`sidebar-item ${activePage === 'visualizer' ? 'active' : ''}`} onClick={() => setActivePage('visualizer')}>
            <Layers size={16} /> 💾 Storage Viz
          </li>
          <li className={`sidebar-item ${activePage === 'performance' ? 'active' : ''}`} onClick={() => setActivePage('performance')}>
            <Activity size={16} /> 📈 Performance
          </li>
          <li className={`sidebar-item ${activePage === 'benchmarks' ? 'active' : ''}`} onClick={() => setActivePage('benchmarks')}>
            <BarChart2 size={16} /> 📊 Benchmarks
          </li>
          <li className={`sidebar-item ${activePage === 'logs' ? 'active' : ''}`} onClick={() => setActivePage('logs')}>
            <Terminal size={16} /> 📝 Logs
          </li>
          <li className={`sidebar-item ${activePage === 'settings' ? 'active' : ''}`} onClick={() => setActivePage('settings')}>
            <Settings size={16} /> ⚙ Settings
          </li>
        </ul>

        {/* Global Mini status indicator */}
        <div style={{ padding: 16, borderTop: '1px solid var(--border-color)', fontSize: 12, display: 'flex', flexDirection: 'column', gap: 6 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: 'var(--text-muted)' }}>Status:</span>
            <span className="metric-badge" style={{ 
              borderColor: systemStats.status === 'Loaded' ? 'var(--neon-green)' : systemStats.status === 'Generating' ? 'var(--neon-cyan)' : 'var(--neon-orange)',
              color: systemStats.status === 'Loaded' ? 'var(--neon-green)' : systemStats.status === 'Generating' ? 'var(--neon-cyan)' : 'var(--neon-orange)'
            }}>
              {systemStats.status}
            </span>
          </div>
          {activeModel && (
            <div style={{ textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap', color: 'var(--text-muted)' }}>
              Model: <span style={{ color: '#fff', fontWeight: 600 }}>{activeModel}</span>
            </div>
          )}
        </div>
      </div>

      {/* MAIN CONTAINER */}
      <div className="main-content">
        {/* HEADER */}
        <header className="header-bar">
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <h2 style={{ fontSize: 18, fontWeight: 700 }}>
              {activePage.charAt(0).toUpperCase() + activePage.slice(1)} Dashboard
            </h2>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            {activeModel ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className="metric-badge" style={{ borderColor: 'var(--neon-green)', color: 'var(--neon-green)' }}>
                  Active VRAM: {systemStats.vram_allocated_mb.toFixed(0)} MB
                </span>
                <button className="danger" onClick={handleUnloadModel} style={{ padding: '6px 12px', fontSize: 13 }}>
                  Unload
                </button>
              </div>
            ) : (
              <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>No model loaded</span>
            )}
          </div>
        </header>

        {/* DYNAMIC PAGE CONTENTS */}
        <main className="page-container">
          
          {/* CHAT PAGE */}
          {activePage === 'chat' && (
            <div className="chat-container">
              <div className="chat-history">
                <div className="messages-list">
                  {messages.map((m, idx) => (
                    <div key={idx} className={`message-wrapper ${m.role}`} style={{ display: 'flex', gap: '16px', maxWidth: '85%', alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start', flexDirection: m.role === 'user' ? 'row-reverse' : 'row', marginBottom: '14px' }}>
                      <div className={`message-avatar ${m.role}`}>
                        {m.role === 'user' ? 'U' : 'T'}
                      </div>
                      <div className="message-bubble" style={{ position: 'relative', display: 'flex', flexDirection: 'column' }}>
                        {m.thinkingContent && (
                          <div style={{ marginBottom: 8, width: '100%', minWidth: '280px' }}>
                            <details style={{ 
                              border: '1px solid var(--border-slate)', 
                              borderRadius: '6px', 
                              background: 'rgba(0,0,0,0.15)',
                              overflow: 'hidden'
                            }}>
                              <summary style={{ 
                                padding: '6px 12px', 
                                fontSize: '12px', 
                                fontWeight: 600, 
                                color: 'var(--accent-purple)', 
                                cursor: 'pointer',
                                userSelect: 'none'
                              }}>
                                Thinking Process
                              </summary>
                              <div style={{ 
                                padding: '10px 12px', 
                                fontSize: '12px', 
                                color: 'var(--text-muted)', 
                                borderTop: '1px solid var(--border-slate)', 
                                whiteSpace: 'pre-wrap',
                                fontFamily: 'Fira Code, monospace'
                              }}>
                                {m.thinkingContent}
                              </div>
                            </details>
                          </div>
                        )}
                        <div>{m.content}</div>

                        {/* Copy button for assistant messages */}
                        {m.role === 'assistant' && (
                          <button 
                            onClick={() => navigator.clipboard.writeText(m.content)}
                            style={{ 
                              alignSelf: 'flex-start',
                              marginTop: '8px',
                              padding: '2px 8px', 
                              fontSize: '10px', 
                              border: '1px solid var(--border-slate)', 
                              background: 'rgba(255,255,255,0.02)', 
                              color: 'var(--text-muted)',
                              borderRadius: '4px',
                              cursor: 'pointer' 
                            }}
                            title="Copy message text"
                          >
                            Copy
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                  
                  {/* Stream rendering */}
                  {isGenerating && (generationOutput || thinkingOutput) && (
                    <div className="message-wrapper assistant" style={{ display: 'flex', gap: '16px', maxWidth: '85%', alignSelf: 'flex-start', flexDirection: 'row' }}>
                      <div className="message-avatar assistant">T</div>
                      <div className="message-bubble" style={{ width: '100%', minWidth: '280px' }}>
                        {thinkingOutput && (
                          <div style={{ marginBottom: 8 }}>
                            <details open style={{ 
                              border: '1px solid var(--border-slate)', 
                              borderRadius: '6px', 
                              background: 'rgba(0,0,0,0.15)',
                              overflow: 'hidden'
                            }}>
                              <summary style={{ 
                                padding: '6px 12px', 
                                fontSize: '12px', 
                                fontWeight: 600, 
                                color: 'var(--accent-purple)', 
                                cursor: 'pointer',
                                userSelect: 'none'
                              }}>
                                Thinking...
                              </summary>
                              <div style={{ 
                                padding: '10px 12px', 
                                fontSize: '12px', 
                                color: 'var(--text-muted)', 
                                borderTop: '1px solid var(--border-slate)', 
                                whiteSpace: 'pre-wrap',
                                fontFamily: 'Fira Code, monospace'
                              }}>
                                {thinkingOutput}
                              </div>
                            </details>
                          </div>
                        )}
                        <div>
                          {generationOutput}
                          <span style={{ display: 'inline-block', width: 6, height: 14, background: 'var(--accent-color)', marginLeft: 4, verticalAlign: 'middle', animation: 'flow-active 1s infinite' }}></span>
                        </div>
                      </div>
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>
 
                <div className="chat-input-area">
                  <textarea 
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    placeholder="Type your prompt here..."
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleSendMessage();
                      }
                    }}
                  />
                  {isGenerating ? (
                    <button className="danger" onClick={handleCancelGeneration}>
                      Cancel
                    </button>
                  ) : (
                    <button className="primary" onClick={handleSendMessage}>
                      <Play size={16} /> Send
                    </button>
                  )}
                </div>
              </div>

              {/* Chat Parameters sidebar */}
              <div className="chat-params-panel glass-card">
                <h3 style={{ fontSize: 14, fontWeight: 700, borderBottom: '1px solid var(--border-color)', paddingBottom: 8, marginBottom: 12 }}>
                  Parameters
                </h3>

                <div className="param-group">
                  <div className="param-header">
                    <span>Temperature</span>
                    <span className="param-value">{engineSettings.runtime.temperature}</span>
                  </div>
                  <input 
                    type="range" min="0" max="1.5" step="0.05"
                    value={engineSettings.runtime.temperature}
                    onChange={(e) => setEngineSettings(prev => ({
                      ...prev,
                      runtime: { ...prev.runtime, temperature: parseFloat(e.target.value) }
                    }))}
                  />
                </div>

                <div className="param-group">
                  <div className="param-header">
                    <span>Top-P</span>
                    <span className="param-value">{engineSettings.runtime.top_p}</span>
                  </div>
                  <input 
                    type="range" min="0" max="1" step="0.01"
                    value={engineSettings.runtime.top_p}
                    onChange={(e) => setEngineSettings(prev => ({
                      ...prev,
                      runtime: { ...prev.runtime, top_p: parseFloat(e.target.value) }
                    }))}
                  />
                </div>

                <div className="param-group">
                  <div className="param-header">
                    <span>Max Tokens</span>
                    <span className="param-value">{engineSettings.runtime.max_new_tokens}</span>
                  </div>
                  <input 
                    type="range" min="1" max="1024" step="8"
                    value={engineSettings.runtime.max_new_tokens}
                    onChange={(e) => setEngineSettings(prev => ({
                      ...prev,
                      runtime: { ...prev.runtime, max_new_tokens: parseInt(e.target.value) }
                    }))}
                  />
                </div>

                <div className="param-group">
                  <div className="param-header">
                    <span>System Prompt</span>
                  </div>
                  <textarea 
                    value={systemPrompt}
                    onChange={(e) => setSystemPrompt(e.target.value)}
                    placeholder="Enter system prompt here..."
                    style={{ height: '60px', fontSize: '12px', resize: 'none', background: 'rgba(0,0,0,0.2)' }}
                  />
                </div>

                <div className="toggle-group" style={{ borderTop: '1px solid var(--border-color)', paddingTop: 12 }}>
                  <span className="toggle-label">Streaming</span>
                  <label className="switch">
                    <input type="checkbox" checked={streamingParam} onChange={(e) => setStreamingParam(e.target.checked)} />
                    <span className="slider"></span>
                  </label>
                </div>

                <div className="toggle-group">
                  <span className="toggle-label">Thinking</span>
                  <label className="switch">
                    <input type="checkbox" checked={thinkingParam === 'on'} onChange={(e) => setThinkingParam(e.target.checked ? 'on' : 'off')} />
                    <span className="slider"></span>
                  </label>
                </div>

                {systemStats.last_metrics && systemStats.last_metrics.tps && (
                  <div style={{ marginTop: 'auto', background: 'rgba(0,0,0,0.2)', padding: 12, borderRadius: 8, fontSize: 12, border: '1px solid var(--border-color)' }}>
                    <div style={{ fontWeight: 700, marginBottom: 4, color: 'var(--neon-cyan)' }}>Last Token Generation</div>
                    <div>Tokens Generated: {systemStats.last_metrics.step}</div>
                    <div>Overall Speed: {systemStats.last_metrics.tps.toFixed(2)} tok/s</div>
                    <div>TTFT: {systemStats.last_metrics.ttft_ms ? `${systemStats.last_metrics.ttft_ms.toFixed(1)} ms` : 'N/A'}</div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* MODELS PAGE */}
          {activePage === 'models' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
              {/* Downloader panel */}
              <div className="glass-card" style={{ padding: 24, display: 'flex', gap: 16, alignItems: 'center' }}>
                <div style={{ flexGrow: 1, display: 'flex', flexDirection: 'column', gap: 6 }}>
                  <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-muted)' }}>Download Model (from Hugging Face)</span>
                  <input 
                    type="text" 
                    placeholder="e.g. Qwen/Qwen3.6-35B-A3B-FP8" 
                    value={downloadModelId}
                    onChange={(e) => setDownloadModelId(e.target.value)}
                    style={{ background: 'rgba(0,0,0,0.4)' }}
                  />
                </div>
                <button className="primary" onClick={handleDownloadModel} style={{ alignSelf: 'flex-end', height: 38 }}>
                  <Download size={16} /> Download
                </button>
              </div>

              {/* Installed / Available lists */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                <h3 style={{ fontSize: 16, fontWeight: 700 }}>Installed and Available Models</h3>
                
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
                  {models.map((model) => (
                    <div key={model.id} className="glass-card" style={{ padding: 20, display: 'flex', flexDirection: 'column', gap: 12, position: 'relative', overflow: 'hidden' }}>
                      {model.status === 'Running' && (
                        <div style={{ position: 'absolute', top: 0, right: 0, background: 'var(--neon-green)', color: '#000', fontSize: 10, fontWeight: 800, padding: '4px 12px', borderBottomLeftRadius: 8 }}>
                          RUNNING
                        </div>
                      )}
                      
                      <div>
                        <h4 style={{ fontSize: 15, fontWeight: 800, color: '#fff' }}>{model.id}</h4>
                        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
                          Location: {model.path || "Hugging Face Hub (Un-downloaded)"}
                        </p>
                      </div>

                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 12, background: 'rgba(0,0,0,0.2)', padding: 10, borderRadius: 6 }}>
                        <div>Architecture: <span style={{ color: '#fff', fontWeight: 600 }}>{model.architecture}</span></div>
                        <div>Quantization: <span style={{ color: '#fff', fontWeight: 600 }}>FP8</span></div>
                        <div>Size on disk: <span style={{ color: '#fff', fontWeight: 600 }}>{model.size_gb ? `${model.size_gb.toFixed(1)} GB` : 'N/A'}</span></div>
                        <div>Layers count: <span style={{ color: '#fff', fontWeight: 600 }}>32</span></div>
                      </div>

                      <div style={{ display: 'flex', gap: 10, marginTop: 8 }}>
                        {model.status === 'Available' ? (
                          <button className="primary" onClick={() => { setDownloadModelId(model.id); handleDownloadModel(); }}>
                            <Download size={14} /> Download
                          </button>
                        ) : model.status === 'Running' ? (
                          <button className="danger" onClick={handleUnloadModel}>
                            Unload
                          </button>
                        ) : (
                          <button className="primary" onClick={() => handleLoadModel(model.id)}>
                            Load Model
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* ENGINE SETTINGS PAGE */}
          {activePage === 'engine' && (
            <div className="glass-card" style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 20 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: 12 }}>
                <h3 style={{ fontSize: 16, fontWeight: 700 }}>Engine Settings</h3>
                <button className="primary" onClick={() => handleSaveSettings(engineSettings)}>
                  Save Configuration
                </button>
              </div>

              {activeModel && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: 12, background: 'rgba(245, 158, 11, 0.1)', border: '1px solid rgba(245, 158, 11, 0.2)', borderRadius: 6, fontSize: 13, color: 'var(--accent-orange)' }}>
                  <AlertCircle size={16} />
                  <span>Note: Changes to precision, memory pool, and cache limits require reloading the active model to take effect.</span>
                  <button className="primary" onClick={() => handleLoadModel(activeModel)} style={{ padding: '4px 10px', fontSize: 11, marginLeft: 'auto' }}>
                    Reload Model
                  </button>
                </div>
              )}

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
                {/* Execution tab */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                  <h4 style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: 6, color: 'var(--neon-cyan)' }}>Execution</h4>
                  
                  <div className="param-group">
                    <label style={{ fontSize: 13, fontWeight: 600 }}>Chunk Size</label>
                    <input type="number" value={32} readOnly />
                  </div>

                  <div className="param-group">
                    <label style={{ fontSize: 13, fontWeight: 600 }}>DType Precision</label>
                    <select 
                      value={engineSettings.execution.dtype} 
                      onChange={(e) => setEngineSettings(prev => ({ 
                        ...prev, 
                        execution: { ...prev.execution, dtype: e.target.value } 
                      }))}
                    >
                      <option value="fp16">FP16</option>
                      <option value="bf16">BF16</option>
                      <option value="fp8">FP8 (Dequant on GPU)</option>
                    </select>
                  </div>

                  <div className="toggle-group">
                    <span className="toggle-label">Active GPU Slot Pinning</span>
                    <label className="switch">
                      <input type="checkbox" defaultChecked />
                      <span className="slider"></span>
                    </label>
                  </div>

                  <div className="toggle-group">
                    <span className="toggle-label">Expert Fusion Execution</span>
                    <label className="switch">
                      <input type="checkbox" defaultChecked />
                      <span className="slider"></span>
                    </label>
                  </div>
                </div>

                {/* Storage & Caches tab */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                  <h4 style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: 6, color: 'var(--neon-purple)' }}>Cache & Storage</h4>
                  
                  <div className="param-group">
                    <label style={{ fontSize: 13, fontWeight: 600 }}>Max GPU Memory Pool (MB)</label>
                    <input 
                      type="number" 
                      value={engineSettings.memory.max_vram_mb} 
                      onChange={(e) => setEngineSettings(prev => ({
                        ...prev,
                        memory: { ...prev.memory, max_vram_mb: parseInt(e.target.value) }
                      }))}
                    />
                  </div>

                  <div className="param-group">
                    <label style={{ fontSize: 13, fontWeight: 600 }}>Max RAM Cache (Percent)</label>
                    <input 
                      type="number" 
                      value={engineSettings.memory.max_ram_percent} 
                      onChange={(e) => setEngineSettings(prev => ({
                        ...prev,
                        memory: { ...prev.memory, max_ram_percent: parseInt(e.target.value) }
                      }))}
                    />
                  </div>

                  <div className="param-group">
                    <label style={{ fontSize: 13, fontWeight: 600 }}>Expert Cache Limit</label>
                    <select 
                      value={engineSettings.cache.expert_limit}
                      onChange={(e) => setEngineSettings(prev => ({
                        ...prev,
                        cache: { ...prev.cache, expert_limit: e.target.value }
                      }))}
                    >
                      <option value="auto">Auto-Scale (VRAM Dependent)</option>
                      <option value="64">64 experts</option>
                      <option value="128">128 experts</option>
                      <option value="256">256 experts</option>
                    </select>
                  </div>

                  <div className="toggle-group">
                    <span className="toggle-label">Memory Mapping (mmap)</span>
                    <label className="switch">
                      <input type="checkbox" defaultChecked />
                      <span className="slider"></span>
                    </label>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* KV CACHE PAGE */}
          {activePage === 'kv' && (
            <div className="glass-card" style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 20 }}>
              <h3 style={{ fontSize: 16, fontWeight: 700, borderBottom: '1px solid var(--border-color)', paddingBottom: 12 }}>
                KV Cache Management
              </h3>
              
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                  <div className="param-group">
                    <label style={{ fontSize: 13, fontWeight: 600 }}>Context Window Length</label>
                    <select>
                      <option>4096 tokens</option>
                      <option>8192 tokens</option>
                      <option>16384 tokens</option>
                    </select>
                  </div>

                  <div className="toggle-group">
                    <span className="toggle-label">PagedAttention Allocation</span>
                    <label className="switch">
                      <input type="checkbox" defaultChecked />
                      <span className="slider"></span>
                    </label>
                  </div>

                  <div className="param-group">
                    <label style={{ fontSize: 13, fontWeight: 600 }}>Paged Block Size</label>
                    <select>
                      <option>16</option>
                      <option>32</option>
                    </select>
                  </div>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                  <div className="toggle-group">
                    <span className="toggle-label">8-bit Cache Values Compression</span>
                    <label className="switch">
                      <input type="checkbox" />
                      <span className="slider"></span>
                    </label>
                  </div>

                  <div className="toggle-group">
                    <span className="toggle-label">Radix Prefix Cache Reuse</span>
                    <label className="switch">
                      <input type="checkbox" defaultChecked />
                      <span className="slider"></span>
                    </label>
                  </div>

                  <div className="toggle-group">
                    <span className="toggle-label">Persistent Disk Cache</span>
                    <label className="switch">
                      <input type="checkbox" />
                      <span className="slider"></span>
                    </label>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* MOE PAGE */}
          {activePage === 'moe' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
              <div className="glass-card" style={{ padding: 20 }}>
                <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 12 }}>Router and Experts Status</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
                  <div style={{ padding: 12, background: 'rgba(0,0,0,0.2)', borderRadius: 8 }}>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Router Strategy</div>
                    <div style={{ fontSize: 18, fontWeight: 800, color: 'var(--neon-cyan)', marginTop: 4 }}>Top-2 Gating</div>
                  </div>
                  <div style={{ padding: 12, background: 'rgba(0,0,0,0.2)', borderRadius: 8 }}>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Active Experts cache</div>
                    <div style={{ fontSize: 18, fontWeight: 800, color: 'var(--neon-green)', marginTop: 4 }}>
                      {systemStats.cache_count} / {systemStats.cache_limit || 128}
                    </div>
                  </div>
                  <div style={{ padding: 12, background: 'rgba(0,0,0,0.2)', borderRadius: 8 }}>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>GPU Cache Hit Rate</div>
                    <div style={{ fontSize: 18, fontWeight: 800, color: 'var(--neon-cyan)', marginTop: 4 }}>
                      {systemStats.last_metrics && systemStats.last_metrics.gpu_hit_rate ? `${systemStats.last_metrics.gpu_hit_rate.toFixed(0)}%` : '0%'}
                    </div>
                  </div>
                  <div style={{ padding: 12, background: 'rgba(0,0,0,0.2)', borderRadius: 8 }}>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>SSD Hit Rate</div>
                    <div style={{ fontSize: 18, fontWeight: 800, color: 'var(--neon-orange)', marginTop: 4 }}>
                      {systemStats.last_metrics && systemStats.last_metrics.ssd_hit_rate ? `${systemStats.last_metrics.ssd_hit_rate.toFixed(0)}%` : '0%'}
                    </div>
                  </div>
                </div>
              </div>

              {/* Active Experts in Slot Buffers */}
              <div className="glass-card" style={{ padding: 20 }}>
                <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 12 }}>GPU Slot Residency (Active Caching)</h3>
                
                {expertStats.length === 0 ? (
                  <div style={{ padding: 30, textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
                    No experts in cache. Trigger generation or load a model to cache weights.
                  </div>
                ) : (
                  <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                      <thead>
                        <tr style={{ borderBottom: '1px solid var(--border-color)', textAlign: 'left', color: 'var(--text-muted)' }}>
                          <th style={{ padding: 10 }}>Layer ID</th>
                          <th style={{ padding: 10 }}>Expert ID</th>
                          <th style={{ padding: 10 }}>Load Speed (Dequant)</th>
                          <th style={{ padding: 10 }}>VRAM Size</th>
                          <th style={{ padding: 10 }}>Total Invocations</th>
                        </tr>
                      </thead>
                      <tbody>
                        {expertStats.map((exp, idx) => (
                          <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                            <td style={{ padding: 10, fontFamily: 'Fira Code', fontWeight: 600 }}>{exp.layer_id}</td>
                            <td style={{ padding: 10, fontFamily: 'Fira Code' }}>
                              <span className="metric-badge" style={{ borderColor: 'var(--neon-cyan)', color: 'var(--neon-cyan)' }}>
                                Expert #{exp.expert_id}
                              </span>
                            </td>
                            <td style={{ padding: 10 }}>{exp.load_ms} ms</td>
                            <td style={{ padding: 10 }}>{exp.vram_cost_kb ? `${(exp.vram_cost_kb / 1024).toFixed(1)} MB` : '9 MB'}</td>
                            <td style={{ padding: 10, color: 'var(--neon-green)', fontWeight: 700 }}>{exp.hits} hits</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* VISUALIZER PAGE */}
          {activePage === 'visualizer' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
              {/* Storage tier streaming visualizer */}
              <div className="glass-card" style={{ padding: 24 }}>
                <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 20 }}>Hierarchy Expert Streaming Visualizer</h3>
                
                {/* SVG nodes flow */}
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-around', background: 'rgba(0,0,0,0.3)', padding: 30, borderRadius: 12, border: '1px solid var(--border-color)' }}>
                  <div className="viz-node ssd">
                    <span>SSD Storage</span>
                    <span style={{ fontSize: 10, opacity: 0.8 }}>Weights Index</span>
                  </div>

                  <svg width="120" height="40">
                    <line x1="0" y1="20" x2="120" y2="20" stroke="var(--neon-purple)" strokeWidth="3" className="flow-line" />
                  </svg>

                  <div className="viz-node ram">
                    <span>System RAM</span>
                    <span style={{ fontSize: 10, opacity: 0.8 }}>
                      {systemStats.ram_hits > 0 ? `${((systemStats.ram_hits / (systemStats.ram_hits + systemStats.ssd_hits || 1)) * 100).toFixed(0)}% Hit` : 'RAM Cache'}
                    </span>
                  </div>

                  <svg width="120" height="40">
                    <line x1="0" y1="20" x2="120" y2="20" stroke="var(--neon-orange)" strokeWidth="3" className="flow-line" />
                  </svg>

                  <div className="viz-node vram">
                    <span>VRAM Pool</span>
                    <span style={{ fontSize: 10, opacity: 0.8 }}>Slot Cache</span>
                  </div>

                  <svg width="120" height="40">
                    <line x1="0" y1="20" x2="120" y2="20" stroke="var(--neon-cyan)" strokeWidth="3" className="flow-line" />
                  </svg>

                  <div className="viz-node gpu">
                    <span>GPU Cores</span>
                    <span style={{ fontSize: 10, opacity: 0.8 }}>Compute GEMM</span>
                  </div>
                </div>
              </div>

              {/* Layer timeline bars */}
              <div className="glass-card" style={{ padding: 24 }}>
                <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 12 }}>Layer Execution Timeline (Last Generated Token)</h3>
                <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 20 }}>
                  Click on any layer bar to inspect details (Load, Dequant, GEMM times).
                </p>

                <div className="layer-progress-container">
                  {Array.from({ length: 32 }).map((_, idx) => {
                    const lData = pipelineLayers[idx] || {
                      load_ms: 0,
                      dequant_ms: 0,
                      gemm_ms: 0,
                      evict_ms: 0,
                      experts: []
                    };
                    
                    const total_ms = lData.load_ms + lData.dequant_ms + lData.gemm_ms + lData.evict_ms || 1;
                    
                    const loadPct = (lData.load_ms / total_ms) * 100;
                    const dequantPct = (lData.dequant_ms / total_ms) * 100;
                    const gemmPct = (lData.gemm_ms / total_ms) * 100;
                    const evictPct = (lData.evict_ms / total_ms) * 100;

                    return (
                      <div 
                        key={idx} 
                        className="layer-timeline-item"
                        onClick={() => setSelectedLayerDetail({ layer_id: idx, ...lData })}
                      >
                        <span className="layer-label">Layer {idx}</span>
                        {total_ms > 1 ? (
                          <div className="progress-bar-stacked">
                            {lData.load_ms > 0 && (
                              <div className="progress-segment load" style={{ width: `${loadPct}%` }} data-tooltip={`Load: ${lData.load_ms.toFixed(1)}ms`}></div>
                            )}
                            {lData.dequant_ms > 0 && (
                              <div className="progress-segment dequant" style={{ width: `${dequantPct}%` }} data-tooltip={`Dequant: ${lData.dequant_ms.toFixed(1)}ms`}></div>
                            )}
                            {lData.gemm_ms > 0 && (
                              <div className="progress-segment gemm" style={{ width: `${gemmPct}%` }} data-tooltip={`GEMM: ${lData.gemm_ms.toFixed(1)}ms`}></div>
                            )}
                            {lData.evict_ms > 0 && (
                              <div className="progress-segment evict" style={{ width: `${evictPct}%` }} data-tooltip={`Evict: ${lData.evict_ms.toFixed(1)}ms`}></div>
                            )}
                          </div>
                        ) : (
                          <div style={{ flexGrow: 1, height: 18, background: 'rgba(255,255,255,0.02)', borderRadius: 4 }}></div>
                        )}
                        <span style={{ fontSize: 11, color: 'var(--text-muted)', width: 60, textAlign: 'right', fontFamily: 'Fira Code' }}>
                          {total_ms > 1 ? `${total_ms.toFixed(1)}ms` : '0ms'}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Layer details popup modal */}
              {selectedLayerDetail && (
                <div style={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', background: 'rgba(0,0,0,0.7)', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <div className="glass-card" style={{ padding: 24, width: 400, border: '1px solid var(--neon-cyan)', boxShadow: 'var(--glow-cyan)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: 10, marginBottom: 16 }}>
                      <h4 style={{ fontSize: 16, fontWeight: 800 }}>Layer #{selectedLayerDetail.layer_id} Details</h4>
                      <button onClick={() => setSelectedLayerDetail(null)} style={{ padding: '2px 8px', fontSize: 12 }}>Close</button>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, fontSize: 14 }}>
                      <div>Copy Time (SSD ➔ RAM ➔ GPU): <span style={{ color: 'var(--neon-purple)', fontWeight: 600 }}>{selectedLayerDetail.load_ms.toFixed(2)} ms</span></div>
                      <div>Dequantization Time: <span style={{ color: 'var(--neon-orange)', fontWeight: 600 }}>{selectedLayerDetail.dequant_ms.toFixed(2)} ms</span></div>
                      <div>GEMM Matrix Mult Time: <span style={{ color: 'var(--neon-cyan)', fontWeight: 600 }}>{selectedLayerDetail.gemm_ms.toFixed(2)} ms</span></div>
                      <div>Eviction / Cache Update: <span style={{ color: 'var(--neon-red)', fontWeight: 600 }}>{selectedLayerDetail.evict_ms.toFixed(2)} ms</span></div>
                      
                      {selectedLayerDetail.experts && selectedLayerDetail.experts.length > 0 && (
                        <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: 10, marginTop: 10 }}>
                          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Routed Experts:</span>
                          <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
                            {selectedLayerDetail.experts.map((exp, i) => (
                              <span key={i} className="metric-badge" style={{ borderColor: 'var(--neon-green)', color: 'var(--neon-green)' }}>
                                Expert #{exp}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* PERFORMANCE PAGE */}
          {activePage === 'performance' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
              {/* Metrics grid */}
              <div className="metrics-grid">
                <div className="metric-card glass-card">
                  <span className="metric-card-title">Token Output Speed</span>
                  <span className="metric-card-value cyan">
                    {systemStats.last_metrics && systemStats.last_metrics.tps ? `${systemStats.last_metrics.tps.toFixed(2)} tok/s` : '0.00 tok/s'}
                  </span>
                  <span className="metric-card-sub">Decode Phase</span>
                </div>
                <div className="metric-card glass-card">
                  <span className="metric-card-title">Peak VRAM Allocated</span>
                  <span className="metric-card-value purple">{systemStats.vram_allocated_mb.toFixed(0)} MB</span>
                  <span className="metric-card-sub">Budget: {systemStats.vram_total_mb.toFixed(0)} MB</span>
                </div>
                <div className="metric-card glass-card">
                  <span className="metric-card-title">System RAM Used</span>
                  <span className="metric-card-value orange">{(systemStats.ram_used_mb / 1024).toFixed(2)} GB</span>
                  <span className="metric-card-sub">{systemStats.ram_usage_percent}% Allocated</span>
                </div>
                <div className="metric-card glass-card">
                  <span className="metric-card-title">CPU Utilization</span>
                  <span className="metric-card-value green">{systemStats.cpu_percent.toFixed(0)}%</span>
                  <span className="metric-card-sub">Host Scheduler</span>
                </div>
              </div>

              {/* Sparkline Charts grid */}
              <div className="charts-container">
                <div className="glass-card" style={{ padding: 20 }}>
                  <h4 style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-muted)', marginBottom: 12, textTransform: 'uppercase' }}>Inference TPS Stream</h4>
                  <Sparkline data={tpsHistory} color="var(--neon-cyan)" maxVal={5} unit="tok/s" />
                </div>
                <div className="glass-card" style={{ padding: 20 }}>
                  <h4 style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-muted)', marginBottom: 12, textTransform: 'uppercase' }}>VRAM Allocated Memory</h4>
                  <Sparkline data={vramHistory} color="var(--neon-purple)" maxVal={6000} unit="MB" />
                </div>
                <div className="glass-card" style={{ padding: 20 }}>
                  <h4 style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-muted)', marginBottom: 12, textTransform: 'uppercase' }}>System RAM Memory</h4>
                  <Sparkline data={ramHistory} color="var(--neon-orange)" maxVal={16384} unit="MB" />
                </div>
                <div className="glass-card" style={{ padding: 20 }}>
                  <h4 style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-muted)', marginBottom: 12, textTransform: 'uppercase' }}>CPU Load Percent</h4>
                  <Sparkline data={cpuHistory} color="var(--neon-green)" maxVal={100} unit="%" />
                </div>
              </div>
            </div>
          )}

          {/* BENCHMARKS PAGE */}
          {activePage === 'benchmarks' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
              <div className="glass-card" style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 16 }}>
                <h3 style={{ fontSize: 16, fontWeight: 700, borderBottom: '1px solid var(--border-color)', paddingBottom: 10 }}>Run Engine Benchmark</h3>
                
                <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 16 }}>
                  <div className="param-group">
                    <label style={{ fontSize: 13, fontWeight: 600 }}>Benchmark Prompt</label>
                    <textarea 
                      value={benchmarkPrompt} 
                      onChange={(e) => setBenchmarkPrompt(e.target.value)} 
                      style={{ height: 60 }} 
                    />
                  </div>
                  <div className="param-group">
                    <label style={{ fontSize: 13, fontWeight: 600 }}>Tokens Limit</label>
                    <input 
                      type="number" 
                      value={benchmarkTokens} 
                      onChange={(e) => setBenchmarkTokens(parseInt(e.target.value))} 
                    />
                  </div>
                </div>

                <button className="primary" onClick={runBenchmark} disabled={isBenchmarking || !activeModel} style={{ alignSelf: 'flex-start' }}>
                  <Play size={14} /> {isBenchmarking ? 'Benchmarking...' : 'Execute Benchmark'}
                </button>
              </div>

              {/* Benchmarks List / Comparisons */}
              <div className="glass-card" style={{ padding: 24 }}>
                <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16 }}>Saved Benchmarks & Comparison</h3>
                
                {pastBenchmarks.length === 0 ? (
                  <div style={{ padding: 30, textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
                    No benchmarks executed yet. Load a model and run a test run to compare results.
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                    <div style={{ overflowX: 'auto' }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                        <thead>
                          <tr style={{ borderBottom: '1px solid var(--border-color)', textAlign: 'left', color: 'var(--text-muted)' }}>
                            <th style={{ padding: 10 }}>Select</th>
                            <th style={{ padding: 10 }}>Model ID</th>
                            <th style={{ padding: 10 }}>Tokens/s</th>
                            <th style={{ padding: 10 }}>TTFT</th>
                            <th style={{ padding: 10 }}>Peak VRAM</th>
                            <th style={{ padding: 10 }}>Peak RAM</th>
                          </tr>
                        </thead>
                        <tbody>
                          {pastBenchmarks.map((bench) => (
                            <tr key={bench.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                              <td style={{ padding: 10 }}>
                                <input 
                                  type="checkbox" 
                                  checked={comparedBenchmarks.includes(bench.id)}
                                  onChange={() => handleBenchmarkCompareToggle(bench.id)}
                                />
                              </td>
                              <td style={{ padding: 10, fontWeight: 700 }}>{bench.model}</td>
                              <td style={{ padding: 10, color: 'var(--neon-green)', fontWeight: 700 }}>{bench.tps} tok/s</td>
                              <td style={{ padding: 10 }}>{bench.ttft_ms} ms</td>
                              <td style={{ padding: 10 }}>{bench.vram_peak_mb.toFixed(0)} MB</td>
                              <td style={{ padding: 10 }}>{(bench.ram_peak_mb / 1024).toFixed(2)} GB</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    {comparedBenchmarks.length > 0 && (
                      <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: 20 }}>
                        <h4 style={{ fontSize: 14, fontWeight: 800, marginBottom: 12, color: 'var(--neon-cyan)' }}>Benchmark Speed Comparison</h4>
                        
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                          {pastBenchmarks.filter(b => comparedBenchmarks.includes(b.id)).map((bench) => (
                            <div key={bench.id} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                              <span style={{ width: 140, textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap', fontSize: 12 }}>{bench.model}</span>
                              <div style={{ flexGrow: 1, background: 'rgba(0,0,0,0.3)', height: 20, borderRadius: 4, overflow: 'hidden' }}>
                                <div style={{ 
                                  width: `${Math.min(100, (bench.tps / 5) * 100)}%`, 
                                  background: 'linear-gradient(95deg, var(--neon-cyan), var(--neon-purple))', 
                                  height: '100%',
                                  transition: 'width 0.4s ease'
                                }}></div>
                              </div>
                              <span style={{ width: 80, fontSize: 13, fontWeight: 700, color: 'var(--neon-green)', fontFamily: 'Fira Code' }}>
                                {bench.tps} TPS
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* LOGS PAGE */}
          {activePage === 'logs' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
              <div style={{ display: 'flex', gap: 10 }}>
                {['all', 'engine', 'storage', 'gpu', 'errors'].map((filter) => (
                  <button 
                    key={filter} 
                    className={selectedLogFilter === filter ? 'primary' : ''}
                    onClick={() => setSelectedLogFilter(filter)}
                    style={{ fontSize: 12, padding: '6px 12px' }}
                  >
                    {filter.toUpperCase()}
                  </button>
                ))}
              </div>

              <div className="terminal-window">
                <div className="terminal-header">
                  <div className="terminal-dots">
                    <span className="terminal-dot red"></span>
                    <span className="terminal-dot yellow"></span>
                    <span className="terminal-dot green"></span>
                  </div>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600 }}>Turbo-LLM System Output</span>
                  <div style={{ width: 36 }}></div>
                </div>

                <div className="terminal-body">
                  {filteredLogs.map((log, idx) => (
                    <div key={idx} className={getLogClass(log)}>
                      {log}
                    </div>
                  ))}
                  <div ref={logsEndRef} />
                </div>
              </div>
            </div>
          )}

          {/* MAIN SETTINGS PAGE */}
          {activePage === 'settings' && (
            <div className="glass-card" style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 20 }}>
              <h3 style={{ fontSize: 16, fontWeight: 700, borderBottom: '1px solid var(--border-color)', paddingBottom: 12 }}>
                Global Application Settings
              </h3>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                <div className="param-group">
                  <label style={{ fontSize: 13, fontWeight: 600 }}>API Server Bind Host</label>
                  <input type="text" defaultValue="127.0.0.1" />
                </div>

                <div className="param-group">
                  <label style={{ fontSize: 13, fontWeight: 600 }}>API Server Port</label>
                  <input type="number" defaultValue="8000" />
                </div>

                <div className="toggle-group">
                  <span className="toggle-label">Start on Host System Boot</span>
                  <label className="switch">
                    <input type="checkbox" />
                    <span className="slider"></span>
                  </label>
                </div>

                <div className="toggle-group">
                  <span className="toggle-label">Enable Hugging Face Cache Sharing</span>
                  <label className="switch">
                    <input type="checkbox" defaultChecked />
                    <span className="slider"></span>
                  </label>
                </div>
              </div>
            </div>
          )}

        </main>
      </div>
    </div>
  );
}
