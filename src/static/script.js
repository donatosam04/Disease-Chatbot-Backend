async function sendMessage() {
    const inputField = document.getElementById("userInput");
    const chatBox = document.getElementById("chatBox");

    const message = inputField.value.trim();
    if (!message) return;

    // Add user message
    addMessage(message, "user");
    inputField.value = "";

    // Show typing indicator
    const typingMessage = addMessage("Typing...", "bot");

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: message })
        });

        const data = await response.json();

        // Remove typing
        typingMessage.remove();

        const reply = data.message || "No response received.";
        const mode = data.mode || "conversation";

        const botMessage = addMessage(reply, "bot");

        // Add mode badge
        const badge = document.createElement("div");
        badge.className = "mode-badge";

        if (mode === "ml_prediction") {
            badge.innerText = "🧠 AI Prediction";
        } else if (mode === "llm_fallback") {
            badge.innerText = "💬 AI Explanation";
        } else {
            badge.innerText = "🤖 Conversation Mode";
        }

        botMessage.appendChild(badge);

    } catch (error) {
        typingMessage.innerText = "Error connecting to server.";
    }

    chatBox.scrollTop = chatBox.scrollHeight;
}


// Unified message function (matches new CSS)
function addMessage(text, sender) {
    const chatBox = document.getElementById("chatBox");

    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", sender);
    messageDiv.innerText = text;

    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;

    return messageDiv;
}


// Handle Enter key
function handleKey(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
}


// Welcome message
window.onload = function () {
    addMessage(
        "Hello! I'm your AI Health Assistant.\nDescribe your symptoms and I will analyze them intelligently.",
        "bot"
    );
};
