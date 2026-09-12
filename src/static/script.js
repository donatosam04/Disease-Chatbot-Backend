const SESSION_ID = crypto.randomUUID();

async function sendMessage() {

    const inputField = document.getElementById("userInput");
    const chatBox = document.getElementById("chatBox");

    const message = inputField.value.trim();
    if (!message) return;

    addMessage(message, "user");
    inputField.value = "";

    const typingMessage = addMessage("Typing...", "bot");

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text: message,
                session_id: SESSION_ID
            })
        });

        const data = await response.json();
        typingMessage.remove();

        const mode = data.mode || "error";
        let reply = "";

        // ----------------------------------
        // HANDLE VACCINE TRACKING
        // ----------------------------------
        if (mode === "vaccine_tracking" && data.data) {

            const age = data.data.child_age;

            reply += `Child Age:\n`;
            reply += `Weeks: ${age.weeks}\n`;
            reply += `Months: ${age.months}\n`;
            reply += `Years: ${age.years}\n\n`;

            reply += `Vaccine Schedule:\n\n`;

            data.data.vaccines.forEach(v => {
                reply += `• ${v.vaccine}\n`;
                reply += `  Prevents: ${v.prevents}\n`;
                reply += `  Scheduled: ${v.scheduled_week} weeks\n`;
                reply += `  Status: ${v.status}\n\n`;
            });
        }

        // ----------------------------------
        // HANDLE EMERGENCY
        // ----------------------------------
        else if (mode === "emergency") {
            reply = data.message;
        }

        // ----------------------------------
        // HANDLE ML PREDICTION
        // ----------------------------------
        else if (mode === "ml_prediction") {
            reply = data.message;
        }

        // ----------------------------------
        // HANDLE CLARIFICATION
        // ----------------------------------
        else if (mode === "clarification") {
            reply = data.message;
        }

        // ----------------------------------
        // HANDLE POST PREDICTION
        // ----------------------------------
        else if (mode === "post_prediction") {
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
            reply = data.message || "System error occurred.";
        }

        // ----------------------------------
        // CATCH-ALL
        // ----------------------------------
        else {
            reply = data.message || "I'm here to help. Please describe your symptoms.";
        }

        const botMessage = addMessage(reply, "bot");

        // ----------------------------------
        // MODE BADGE SYSTEM
        // ----------------------------------

        const badge = document.createElement("div");
        badge.className = "mode-badge";

        if (mode === "emergency") {
            badge.innerText = "🚨 Emergency Alert";
        }
        else if (mode === "ml_prediction") {
            badge.innerText = "🧠 AI Prediction";
        }
        else if (mode === "vaccine_tracking") {
            badge.innerText = "💉 Vaccine Tracker";
        }
        else if (mode === "llm_fallback") {
            badge.innerText = "💬 AI Explanation";
        }
        else if (mode === "clarification") {
            badge.innerText = "🔎 Needs More Information";
        }
        else if (mode === "post_prediction") {
            badge.innerText = "💊 Health Information";
        }
        else {
            badge.innerText = "⚙ System";
        }

        botMessage.appendChild(badge);

    } catch (error) {
        typingMessage.innerText = "Error connecting to server.";
        console.error(error);
    }

    chatBox.scrollTop = chatBox.scrollHeight;
}


function addMessage(text, sender) {

    const chatBox = document.getElementById("chatBox");

    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", sender);

    messageDiv.innerText = text;

    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;

    return messageDiv;
}


function handleKey(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
}


window.onload = function () {
    addMessage(
        "Hello! I'm your AI Health Assistant.\nDescribe your symptoms and I will analyze them intelligently.",
        "bot"
    );
};