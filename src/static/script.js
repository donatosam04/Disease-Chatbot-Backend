async function sendMessage() {

    const inputField = document.getElementById("userInput");
    const chatBox = document.getElementById("chatBox");

    const message = inputField.value.trim();
    if (!message) return;

    // Add user message
    addMessage(message, "user");
    inputField.value = "";

    // Typing indicator
    const typingMessage = addMessage("Typing...", "bot");

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: message })  // ✅ Matches FastAPI
        });

        const data = await response.json();

        typingMessage.remove();

        const mode = data.mode || "conversation";
        let reply = "";

        // ----------------------------------
        // HANDLE VACCINE TRACKING
        // ----------------------------------
        if (mode === "vaccine_tracking" && data.data) {

            const age = data.data.child_age;

            reply += `🧒 Child Age:\n`;
            reply += `Weeks: ${age.weeks}\n`;
            reply += `Months: ${age.months}\n`;
            reply += `Years: ${age.years}\n\n`;

            reply += `💉 Vaccine Schedule:\n\n`;

            data.data.vaccines.forEach(v => {
                reply += `• ${v.vaccine}\n`;
                reply += `  Prevents: ${v.prevents}\n`;
                reply += `  Scheduled: ${v.scheduled_week} weeks\n`;
                reply += `  Status: ${v.status}\n\n`;
            });

        }

        // ----------------------------------
        // HANDLE ML PREDICTION
        // ----------------------------------
        else if (mode === "ml_prediction") {
            reply = data.message;
        }

        // ----------------------------------
        // HANDLE LLM FALLBACK
        // ----------------------------------
        else if (mode === "llm_fallback") {
            reply = data.message;
        }

        // ----------------------------------
        // HANDLE ERROR
        // ----------------------------------
        else if (mode === "error") {
            reply = data.message;
        }

        else {
            reply = "No response received.";
        }

        const botMessage = addMessage(reply, "bot");

        // Mode Badge
        const badge = document.createElement("div");
        badge.className = "mode-badge";

        if (mode === "ml_prediction") {
            badge.innerText = "🧠 AI Prediction";
        } else if (mode === "vaccine_tracking") {
            badge.innerText = "💉 Vaccine Tracker";
        } else if (mode === "llm_fallback") {
            badge.innerText = "💬 AI Explanation";
        } else {
            badge.innerText = "🤖 Conversation Mode";
        }

        botMessage.appendChild(badge);

    } catch (error) {
        typingMessage.innerText = "Error connecting to server.";
        console.error(error);
    }

    chatBox.scrollTop = chatBox.scrollHeight;
}


// Add message to chat
function addMessage(text, sender) {

    const chatBox = document.getElementById("chatBox");

    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", sender);

    messageDiv.innerText = text;

    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;

    return messageDiv;
}


// Enter key support
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