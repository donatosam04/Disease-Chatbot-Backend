// Global Configuration
const CONFIG = {
  BACKEND_URL: 'https://your-backend.onrender.com',  // Node/Express API
  AI_URL: 'https://your-ai-gateway.onrender.com',    // FastAPI AI Gateway
};

// --- AUTH & INIT ---
document.addEventListener('DOMContentLoaded', () => {
  // Check if we are on index.html or app.html
  const isAuthPage = document.querySelector('.auth-container') !== null;
  const token = localStorage.getItem('jwt_token');

  if (isAuthPage) {
    if (token) {
      window.location.href = 'app.html';
    }
    
    // Auth Event Listeners
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');
    
    if (loginForm) {
      loginForm.addEventListener('submit', handleLogin);
    }
    if (registerForm) {
      registerForm.addEventListener('submit', handleRegister);
    }
  } else {
    // We are on app.html
    if (!token) {
      window.location.href = 'index.html';
      return;
    }
    
    // Load User Data
    const userData = JSON.parse(localStorage.getItem('user_data') || '{}');
    document.getElementById('user-name').textContent = userData.name || 'User';
    document.getElementById('user-details').textContent = `${userData.age || '--'} yrs · ${userData.gender || '--'} · ${userData.city || '--'}`;
    if (userData.name) {
      document.getElementById('user-avatar').textContent = userData.name.charAt(0).toUpperCase();
    }
    
    document.getElementById('vax-user-name').textContent = `👤 ${userData.name || 'User'} (${userData.age || '--'} yrs)`;

    // App Event Listeners
    document.getElementById('chat-form').addEventListener('submit', handleChatSubmit);
    
    // Load initial views
    loadChatHistory();
  }
});

// --- AUTH LOGIC ---
function switchTab(tab) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.auth-form').forEach(form => form.classList.remove('active'));
  
  document.getElementById(`tab-${tab}`).classList.add('active');
  document.getElementById(`${tab}-form`).classList.add('active');
}

async function handleLogin(e) {
  e.preventDefault();
  const email = document.getElementById('login-email').value;
  const password = document.getElementById('login-password').value;
  
  const btn = e.target.querySelector('button');
  const originalText = btn.innerHTML;
  btn.innerHTML = 'Loading...';
  
  try {
    const res = await fetch(`${CONFIG.BACKEND_URL}/api/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    });
    
    if (res.ok) {
      const data = await res.json();
      localStorage.setItem('jwt_token', data.token || 'mock_token_123');
      localStorage.setItem('user_data', JSON.stringify(data.user || { name: 'Arunya', age: 22, gender: 'Male', city: 'Coimbatore' }));
      showToast('Login successful!', 'success');
      setTimeout(() => window.location.href = 'app.html', 1000);
    } else {
      // Mock success for development if backend fails
      console.warn("Backend failed, mocking login for local testing.");
      localStorage.setItem('jwt_token', 'mock_token_123');
      localStorage.setItem('user_data', JSON.stringify({ name: email.split('@')[0], age: 25, gender: 'Other', city: 'Unknown' }));
      showToast('Mock login successful!', 'success');
      setTimeout(() => window.location.href = 'app.html', 1000);
    }
  } catch (error) {
    console.error(error);
    // Mock success for development
    localStorage.setItem('jwt_token', 'mock_token_123');
    localStorage.setItem('user_data', JSON.stringify({ name: email.split('@')[0], age: 25, gender: 'Other', city: 'Unknown' }));
    showToast('Mock login successful (network error)!', 'success');
    setTimeout(() => window.location.href = 'app.html', 1000);
  } finally {
    btn.innerHTML = originalText;
  }
}

async function handleRegister(e) {
  e.preventDefault();
  const name = document.getElementById('reg-name').value;
  const email = document.getElementById('reg-email').value;
  const password = document.getElementById('reg-password').value;
  const age = document.getElementById('reg-age').value;
  const gender = document.getElementById('reg-gender').value;
  const city = document.getElementById('reg-city').value;
  
  const btn = e.target.querySelector('button');
  btn.innerHTML = 'Registering...';
  
  try {
    const res = await fetch(`${CONFIG.BACKEND_URL}/api/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, email, password, age, gender, city })
    });
    
    if(res.ok) {
      showToast('Registration successful! Please login.', 'success');
      switchTab('login');
    } else {
      showToast('Mock Registration complete! Please login.', 'success');
      switchTab('login');
    }
  } catch(error) {
    showToast('Mock Registration complete! Please login.', 'success');
    switchTab('login');
  } finally {
    btn.innerHTML = 'Register &rarr;';
  }
}

function logout() {
  localStorage.removeItem('jwt_token');
  localStorage.removeItem('user_data');
  window.location.href = 'index.html';
}

// --- APP LOGIC ---
function switchView(viewName) {
  document.querySelectorAll('.view-section').forEach(el => el.style.display = 'none');
  document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
  
  document.getElementById(`view-${viewName}`).style.display = 'flex';
  document.getElementById(`nav-${viewName === 'vaccines' ? 'vaccines' : 'chat'}`).classList.add('active');
  
  if (viewName === 'vaccines') {
    loadVaccines();
  }
}

// --- CHAT LOGIC ---
const chatMessages = document.getElementById('chat-messages');

function startNewChat() {
  if(!chatMessages) return;
  chatMessages.innerHTML = '';
  appendBotMessage("Hello. How are you feeling today? Do you have any health concerns or symptoms you'd like to talk about?", 'CHITCHAT');
}

async function loadChatHistory() {
  if(!chatMessages) return;
  startNewChat(); // Default state
  
  try {
    const token = localStorage.getItem('jwt_token');
    const res = await fetch(`${CONFIG.BACKEND_URL}/api/chat-history`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    // If backend is real, populate history here
  } catch(e) {
    console.log("Using default chat state");
  }
}

async function handleChatSubmit(e) {
  e.preventDefault();
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if(!text) return;
  
  appendUserMessage(text);
  input.value = '';
  
  showTypingIndicator();
  
  // Call AI Backend
  try {
    const token = localStorage.getItem('jwt_token');
    const res = await fetch(`${CONFIG.AI_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({ message: text })
    });
    
    removeTypingIndicator();
    
    if (res.ok) {
      const data = await res.json();
      // Handle actual AI response format
      renderAIResponse(data);
    } else {
      mockAIResponse(text);
    }
  } catch (error) {
    removeTypingIndicator();
    mockAIResponse(text);
  }
}

// UI Helpers for Chat
function appendUserMessage(text) {
  const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
  const html = `
    <div class="message user">
      <div class="bubble">${text}</div>
      <div class="timestamp">${time}</div>
    </div>
  `;
  chatMessages.insertAdjacentHTML('beforeend', html);
  scrollToBottom();
}

function appendBotMessage(text, label = 'CHITCHAT') {
  const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
  const html = `
    <div class="message bot">
      <div class="message-label">• ${label}</div>
      <div class="bubble">${text}</div>
      <div class="timestamp">${time}</div>
    </div>
  `;
  chatMessages.insertAdjacentHTML('beforeend', html);
  scrollToBottom();
}

function renderMLPredictionCard(diseaseName, confidence, emoji = '🤒') {
  const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
  const confPercent = Math.round(confidence * 100);
  const isHighConf = confPercent > 60;
  
  let actionsHtml = '';
  if (isHighConf) {
    actionsHtml = `
      <div class="action-buttons">
        <button class="btn-sm" onclick="showGuidance('danger')">⚠ Is it dangerous?</button>
        <button class="btn-sm" onclick="showGuidance('treatment')">💊 Treatment</button>
        <button class="btn-sm" onclick="showGuidance('causes')">📋 What causes this?</button>
      </div>
    `;
  } else {
    actionsHtml = `<p style="font-size: 0.85rem; color: var(--text-secondary);">I need more details. Could you describe your symptoms further?</p>`;
  }

  const html = `
    <div class="message bot ml-card">
      <div class="message-label" style="color: var(--accent-cyan);">• ML PREDICTION</div>
      <div class="bubble" style="background: transparent; border: none; padding: 0 16px 16px;">
        <div class="disease-title">${emoji} ${diseaseName}</div>
        <div class="confidence-section">
          <div class="conf-label"><span>AI Confidence</span> <span>${confPercent}%</span></div>
          <div class="progress-bar">
            <div class="progress-fill" style="width: ${confPercent}%;"></div>
          </div>
        </div>
        <div class="disclaimer">⚠ AI-based prediction. Please consult a medical professional for accurate diagnosis.</div>
        ${actionsHtml}
      </div>
      <div class="timestamp">${time}</div>
    </div>
  `;
  chatMessages.insertAdjacentHTML('beforeend', html);
  scrollToBottom();
}

function showGuidance(type) {
  const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
  
  let content = '';
  if (type === 'treatment') {
    content = `
      <div class="warning-block">Severity: Moderate. Over-the-counter medication may help, but if symptoms persist for >3 days, visit the nearest clinic.</div>
      <div class="clinical-detail">
        <h5>• AI CLINICAL DETAIL</h5>
        <p>Recommended next steps:</p>
        <ul>
          <li>Rest and stay hydrated</li>
          <li>Take paracetamol for fever</li>
          <li>Differential Diagnosis: Could also be seasonal flu</li>
        </ul>
      </div>
    `;
  } else {
    content = `<p>Detailed guidance for ${type} is not available in mock mode.</p>`;
  }
  
  const html = `
    <div class="message bot guidance-card">
      <div class="message-label" style="color: var(--info-blue);">• HEALTH GUIDANCE</div>
      <div class="bubble">
        ${content}
      </div>
      <div class="timestamp">${time}</div>
    </div>
  `;
  chatMessages.insertAdjacentHTML('beforeend', html);
  scrollToBottom();
}

function showTypingIndicator() {
  const html = `
    <div class="message bot" id="typing-indicator">
      <div class="typing-indicator">
        <div class="dot"></div>
        <div class="dot"></div>
        <div class="dot"></div>
      </div>
    </div>
  `;
  chatMessages.insertAdjacentHTML('beforeend', html);
  scrollToBottom();
}

function removeTypingIndicator() {
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
}

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Mock AI Logic for demonstration
function mockAIResponse(text) {
  text = text.toLowerCase();
  setTimeout(() => {
    if (text.includes('fever') && text.includes('headache')) {
      renderMLPredictionCard('Dengue', 0.85, '🦟');
    } else if (text.includes('cold') || text.includes('cough')) {
      renderMLPredictionCard('Common Cold', 0.92, '🤧');
    } else if (text.includes('pain')) {
      renderMLPredictionCard('Unknown Condition', 0.45, '❓');
    } else {
      appendBotMessage("I see. Can you tell me how long you've been experiencing this?", 'CHITCHAT');
    }
  }, 1500);
}


// --- SYMPTOM CHECKER MODAL ---
function openSymptomModal() {
  document.getElementById('symptom-modal').classList.add('active');
}

function closeSymptomModal() {
  document.getElementById('symptom-modal').classList.remove('active');
}

function toggleChip(btn) {
  btn.classList.toggle('selected');
}

function analyzeSymptoms() {
  const text = document.getElementById('symptom-text').value;
  const selectedChips = Array.from(document.querySelectorAll('.chip.selected')).map(c => c.innerText);
  
  let combinedQuery = text;
  if(selectedChips.length > 0) {
    combinedQuery += ` (Symptoms: ${selectedChips.join(', ')})`;
  }
  
  closeSymptomModal();
  switchView('chat');
  
  const input = document.getElementById('chat-input');
  input.value = combinedQuery;
  document.getElementById('chat-form').dispatchEvent(new Event('submit'));
}

// --- VACCINE TRACKER ---
const mockVaccines = [
  { name: 'BCG', desc: 'Tuberculosis', age: 'At birth', status: 'completed', icon: '👶' },
  { name: 'OPV 0', desc: 'Polio', age: 'At birth', status: 'completed', icon: '💧' },
  { name: 'Hepatitis B', desc: 'Hep B Infection', age: 'At birth', status: 'completed', icon: '🩸' },
  { name: 'DPT 1', desc: 'Diphtheria, Pertussis, Tetanus', age: '6 weeks', status: 'completed', icon: '💉' },
  { name: 'Rotavirus 1', desc: 'Diarrhea', age: '6 weeks', status: 'overdue', icon: '🦠' },
  { name: 'PCV 1', desc: 'Pneumonia', age: '6 weeks', status: 'due', icon: '🫁' },
  { name: 'Measles', desc: 'Measles', age: '9 months', status: 'upcoming', icon: '🔴' }
];

async function loadVaccines() {
  // Mock API Call
  renderVaccines(mockVaccines);
}

function renderVaccines(vaccines) {
  const grid = document.getElementById('vaccine-grid');
  if(!grid) return;
  grid.innerHTML = '';
  
  let counts = { completed: 0, due: 0, overdue: 0, upcoming: 0 };
  
  vaccines.forEach(v => {
    counts[v.status]++;
    const html = `
      <div class="vax-card">
        <div class="vax-card-header">
          <div class="vax-icon-name">
            <span class="vax-icon">${v.icon}</span>
            <span class="vax-name">${v.name}</span>
          </div>
          <span class="badge ${v.status}">${v.status.toUpperCase()}</span>
        </div>
        <div class="vax-desc">${v.desc}</div>
        <div class="vax-age">${v.age}</div>
        <div class="vax-actions">
          ${v.status !== 'completed' ? `<button class="btn-outline" onclick="markVaxCompleted('${v.name}')">✓ Completed</button>` : ''}
          ${v.status === 'due' || v.status === 'upcoming' ? `<button class="btn-outline" onclick="remindVax('${v.name}')">🔔 Remind</button>` : ''}
        </div>
      </div>
    `;
    grid.insertAdjacentHTML('beforeend', html);
  });
  
  document.getElementById('vax-count-complete').innerText = counts.completed;
  document.getElementById('vax-count-due').innerText = counts.due;
  document.getElementById('vax-count-overdue').innerText = counts.overdue;
  document.getElementById('vax-count-upcoming').innerText = counts.upcoming;
}

function registerVaccine() {
  showToast('Registration feature coming soon!', 'info');
}

function markVaxCompleted(name) {
  showToast(`Marked ${name} as completed!`, 'success');
}

function remindVax(name) {
  showToast(`Reminder set for ${name}.`, 'info');
}

// --- UTILS ---
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if(!container) return;
  
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerText = message;
  
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}
