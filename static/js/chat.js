/**
 * Client-Side Controller for AI Customer Support Chatbot.
 * Handles instant messaging, contextual multi-turn conversation,
 * interactive card rendering, feedback, and live NLP diagnostics.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const chatContainer = document.getElementById('chatContainer');
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const typingIndicator = document.getElementById('typingIndicator');
    const quickReplies = document.getElementById('quickReplies');
    const toggleInspectorBtn = document.getElementById('toggleInspectorBtn');
    const inspectorPanel = document.getElementById('inspectorPanel');
    const resetChatBtn = document.getElementById('resetChatBtn');

    // Inspector fields
    const sessionDisplay = document.getElementById('sessionDisplay');
    const inspectIntent = document.getElementById('inspectIntent');
    const inspectConfidence = document.getElementById('inspectConfidence');
    const inspectSentiment = document.getElementById('inspectSentiment');
    const inspectState = document.getElementById('inspectState');
    const inspectLatency = document.getElementById('inspectLatency');
    const latencyBadge = document.getElementById('latencyBadge');

    // Session ID management
    let sessionId = localStorage.getItem('chatbot_session_id');
    if (!sessionId) {
        sessionId = 'sess_' + Math.random().toString(36).substring(2, 12);
        localStorage.setItem('chatbot_session_id', sessionId);
    }
    sessionDisplay.textContent = `Session: ${sessionId}`;

    // Set initial timestamp
    const initialTimeEl = document.getElementById('initialTime');
    if (initialTimeEl) {
        initialTimeEl.textContent = formatTime(new Date());
    }

    // Toggle Inspector Panel
    toggleInspectorBtn.addEventListener('click', () => {
        inspectorPanel.classList.toggle('hidden');
    });

    // Reset Chat Context
    resetChatBtn.addEventListener('click', async () => {
        if (!confirm('Are you sure you want to reset the conversational context?')) return;

        try {
            await fetch('/api/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId })
            });

            appendSystemNotice('Conversational context has been reset.');
            inspectState.textContent = 'default';
        } catch (err) {
            console.error('Reset error:', err);
        }
    });

    // Handle Quick Reply Chips
    quickReplies.addEventListener('click', (e) => {
        const chip = e.target.closest('.chip');
        if (chip) {
            const query = chip.getAttribute('data-query');
            if (query) {
                userInput.value = query;
                chatForm.dispatchEvent(new Event('submit'));
            }
        }
    });

    // Handle Form Submission
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const message = userInput.value.trim();
        if (!message) return;

        // 1. Render User Message
        appendMessage('user', message);
        userInput.value = '';
        userInput.focus();

        // 2. Show Typing Indicator
        showTyping(true);

        try {
            const startTime = performance.now();
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: message,
                    session_id: sessionId
                })
            });

            const data = await res.json();
            showTyping(false);

            if (data.success) {
                // 3. Render Bot Response with optional interactive card
                appendMessage('bot', data.response, data.card, data.log_id);

                // 4. Update Inspector Diagnostics
                updateInspector(data);
            } else {
                appendMessage('bot', `I encountered an issue: ${data.error || 'Please try again.'}`);
            }
        } catch (error) {
            showTyping(false);
            console.error('Chat error:', error);
            appendMessage('bot', "I'm having trouble connecting to the service. Please make sure the server is running.");
        }
    });

    // Append Message to UI
    function appendMessage(sender, text, card = null, logId = null) {
        const row = document.createElement('div');
        row.className = `message-row ${sender}-row`;

        const avatar = document.createElement('div');
        avatar.className = 'msg-avatar';
        avatar.innerHTML = sender === 'bot' ? '<i class="fa-solid fa-robot"></i>' : '<i class="fa-solid fa-user"></i>';

        const bubbleWrapper = document.createElement('div');
        bubbleWrapper.className = 'msg-bubble-wrapper';

        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble';
        bubble.innerHTML = parseMarkdown(text);

        // Render Interactive Cards
        if (card) {
            const cardEl = renderCard(card);
            if (cardEl) bubble.appendChild(cardEl);
        }

        bubbleWrapper.appendChild(bubble);

        // Timestamp & Action buttons
        const metaRow = document.createElement('div');
        metaRow.className = 'msg-time';
        metaRow.textContent = formatTime(new Date());
        bubbleWrapper.appendChild(metaRow);

        // Feedback buttons for bot messages
        if (sender === 'bot' && logId) {
            const actions = document.createElement('div');
            actions.className = 'msg-actions';
            actions.innerHTML = `
                <button class="feedback-btn" data-rate="positive" data-log="${logId}" title="Helpful response">
                    <i class="fa-regular fa-thumbs-up"></i> Helpful
                </button>
                <button class="feedback-btn" data-rate="negative" data-log="${logId}" title="Not helpful">
                    <i class="fa-regular fa-thumbs-down"></i>
                </button>
            `;
            bubbleWrapper.appendChild(actions);

            // Add feedback click listeners
            actions.addEventListener('click', async (e) => {
                const btn = e.target.closest('.feedback-btn');
                if (btn) {
                    const rating = btn.getAttribute('data-rate');
                    const targetLogId = btn.getAttribute('data-log');
                    await submitFeedback(targetLogId, rating, btn, actions);
                }
            });
        }

        row.appendChild(avatar);
        row.appendChild(bubbleWrapper);
        chatContainer.appendChild(row);

        scrollToBottom();
    }

    // Interactive Card Widget Renderer
    function renderCard(card) {
        const cardDiv = document.createElement('div');
        cardDiv.className = 'card-widget';

        if (card.type === 'order_status') {
            const statusClass = card.status === 'Delivered' ? 'badge-delivered' :
                               (card.status === 'In Transit' || card.status === 'Out for Delivery' ? 'badge-in-transit' : 'badge-processing');

            cardDiv.innerHTML = `
                <div class="card-widget-header">
                    <span class="title"><i class="fa-solid fa-box-open"></i> Order ${card.order_id}</span>
                    <span class="card-badge ${statusClass}">${card.status}</span>
                </div>
                <div class="card-details-grid">
                    <div class="card-row">
                        <span class="c-label">Customer</span>
                        <span class="c-val">${card.customer || 'Customer'}</span>
                    </div>
                    <div class="card-row">
                        <span class="c-label">Carrier & Tracking</span>
                        <span class="c-val">${card.carrier} (${card.tracking_code})</span>
                    </div>
                    <div class="card-row card-full-width">
                        <span class="c-label">Purchased Items</span>
                        <span class="c-val">${card.items}</span>
                    </div>
                    <div class="card-row">
                        <span class="c-label">Estimated Arrival</span>
                        <span class="c-val">${card.delivery}</span>
                    </div>
                    <div class="card-row">
                        <span class="c-label">Total Amount</span>
                        <span class="c-val">${card.total}</span>
                    </div>
                </div>
            `;
            return cardDiv;
        } else if (card.type === 'return_auth') {
            cardDiv.innerHTML = `
                <div class="card-widget-header">
                    <span class="title"><i class="fa-solid fa-receipt"></i> Return Authorization</span>
                    <span class="card-badge badge-delivered">APPROVED</span>
                </div>
                <div class="card-details-grid">
                    <div class="card-row">
                        <span class="c-label">RMA Code</span>
                        <span class="c-val" style="color:#38bdf8;font-weight:700;">${card.rma_code}</span>
                    </div>
                    <div class="card-row">
                        <span class="c-label">Order Reference</span>
                        <span class="c-val">${card.order_id}</span>
                    </div>
                    <div class="card-row card-full-width">
                        <span class="c-label">Next Steps</span>
                        <span class="c-val">Label sent to your email. Return drop-off at any authorized partner.</span>
                    </div>
                </div>
            `;
            return cardDiv;
        } else if (card.type === 'support_ticket') {
            cardDiv.innerHTML = `
                <div class="card-widget-header">
                    <span class="title"><i class="fa-solid fa-headset"></i> Support Ticket</span>
                    <span class="card-badge badge-processing">${card.priority}</span>
                </div>
                <div class="card-details-grid">
                    <div class="card-row">
                        <span class="c-label">Ticket ID</span>
                        <span class="c-val" style="color:#a78bfa;font-weight:700;">${card.ticket_id}</span>
                    </div>
                    <div class="card-row">
                        <span class="c-label">Status</span>
                        <span class="c-val">Queued for Human Agent</span>
                    </div>
                    <div class="card-row card-full-width">
                        <span class="c-label">Resolution SLA</span>
                        <span class="c-val">Response guaranteed within 2 hours.</span>
                    </div>
                </div>
            `;
            return cardDiv;
        }

        return null;
    }

    // Submit Feedback
    async function submitFeedback(logId, rating, activeBtn, container) {
        try {
            await fetch('/api/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ log_id: logId, rating: rating })
            });

            container.innerHTML = `<span style="font-size:0.75rem; color:#10b981;"><i class="fa-solid fa-check"></i> Thanks for your feedback!</span>`;
        } catch (err) {
            console.error('Feedback error:', err);
        }
    }

    // Append System Notice
    function appendSystemNotice(text) {
        const div = document.createElement('div');
        div.style.cssText = 'text-align:center; margin:8px 0; font-size:0.75rem; color:#94a3b8; font-style:italic;';
        div.innerHTML = `<i class="fa-solid fa-info-circle"></i> ${text}`;
        chatContainer.appendChild(div);
        scrollToBottom();
    }

    // Update NLP Inspector UI
    function updateInspector(data) {
        inspectIntent.textContent = data.intent || 'fallback';
        const confPct = Math.round((data.confidence || 0) * 100);
        inspectConfidence.textContent = `${confPct}%`;
        inspectSentiment.textContent = `${data.sentiment || 'neutral'} (${data.sentiment_score || 0})`;
        inspectState.textContent = data.state || 'default';
        inspectLatency.textContent = `${data.latency_ms} ms`;
        latencyBadge.textContent = `${data.latency_ms} ms`;
    }

    // Helpers
    function showTyping(show) {
        typingIndicator.classList.toggle('hidden', !show);
        if (show) scrollToBottom();
    }

    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function formatTime(date) {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    function parseMarkdown(text) {
        if (!text) return '';
        // Bold **text**
        let formatted = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // Italic *text*
        formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
        // Inline code `code`
        formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
        // Newlines to line breaks
        formatted = formatted.replace(/\n/g, '<br>');
        return formatted;
    }
});
