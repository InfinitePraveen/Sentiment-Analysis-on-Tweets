document.addEventListener('DOMContentLoaded', function () {
    const tweetInput = document.getElementById('tweet-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultSection = document.getElementById('result-section');
    const historySection = document.getElementById('history-section');
    const charCount = document.getElementById('char-count');
    const maxChars = 280;

    // Character counter
    tweetInput.addEventListener('input', function () {
        const length = this.value.length;
        charCount.textContent = length;

        if (length > maxChars) {
            charCount.style.color = '#ff6b6b';
        } else {
            charCount.style.color = '#999';
        }
    });

    // Auto-resize textarea
    tweetInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = this.scrollHeight + 'px';
    });

    // Analyze button click
    analyzeBtn.addEventListener('click', function () {
        const tweet = tweetInput.value.trim();

        if (!tweet) {
            alert('Please enter a tweet to analyze.');
            return;
        }

        if (tweet.length > maxChars) {
            alert(`Tweet exceeds ${maxChars} characters. Please shorten it.`);
            return;
        }

        analyzeTweet(tweet);
    });

    // Enter key to submit
    tweetInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            analyzeBtn.click();
        }
    });

    async function analyzeTweet(tweet) {
        // Show loading state
        analyzeBtn.textContent = '⏳ Analyzing...';
        analyzeBtn.disabled = true;

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ tweet: tweet })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();

            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }

            updateResults(data);

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while analyzing the tweet. Please try again.');
        } finally {
            analyzeBtn.innerHTML = '<span class="btn-icon">🔍</span> Analyze Sentiment';
            analyzeBtn.disabled = false;
        }
    }

    function updateResults(data) {
        const { sentiment, confidence, probabilities, history } = data;

        // Show result section
        resultSection.classList.remove('hidden');

        // Update sentiment icon
        const iconMap = {
            'positive': '😊',
            'negative': '😞',
            'neutral': '😐'
        };
        document.getElementById('sentiment-icon').textContent = iconMap[sentiment] || '🤔';

        // Update sentiment label
        const labelMap = {
            'positive': 'Positive 😊',
            'negative': 'Negative 😞',
            'neutral': 'Neutral 😐'
        };
        document.getElementById('sentiment-label').textContent = labelMap[sentiment] || 'Unknown';

        // Update confidence bar
        const confidenceFill = document.getElementById('confidence-fill');
        confidenceFill.style.width = confidence + '%';
        document.getElementById('confidence-text').textContent = confidence.toFixed(1) + '%';

        // Remove previous classes
        confidenceFill.classList.remove('negative', 'neutral', 'positive');
        confidenceFill.classList.add(sentiment);

        // Update probabilities
        const probNegative = document.getElementById('prob-negative');
        const probNeutral = document.getElementById('prob-neutral');
        const probPositive = document.getElementById('prob-positive');

        probNegative.style.width = probabilities.negative + '%';
        probNeutral.style.width = probabilities.neutral + '%';
        probPositive.style.width = probabilities.positive + '%';

        document.getElementById('prob-negative-text').textContent = probabilities.negative.toFixed(1) + '%';
        document.getElementById('prob-neutral-text').textContent = probabilities.neutral.toFixed(1) + '%';
        document.getElementById('prob-positive-text').textContent = probabilities.positive.toFixed(1) + '%';

        // Update history
        if (history && history.length > 0) {
            historySection.classList.remove('hidden');
            updateHistory(history);
        }
    }

    function updateHistory(history) {
        const historyList = document.getElementById('history-list');
        historyList.innerHTML = '';

        // Show only latest 5
        const recentHistory = history.slice(-5).reverse();

        recentHistory.forEach(item => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';

            const tweetText = document.createElement('span');
            tweetText.className = 'tweet-text';
            tweetText.textContent = item.tweet.length > 60 ?
                item.tweet.substring(0, 60) + '...' :
                item.tweet;

            const badge = document.createElement('span');
            badge.className = `sentiment-badge ${item.sentiment}`;
            badge.textContent = item.sentiment.charAt(0).toUpperCase() + item.sentiment.slice(1);

            const confidenceBadge = document.createElement('span');
            confidenceBadge.className = 'confidence-badge';
            confidenceBadge.textContent = `${item.confidence}%`;

            const timestamp = document.createElement('span');
            timestamp.className = 'timestamp';
            timestamp.textContent = item.timestamp;

            historyItem.appendChild(tweetText);
            historyItem.appendChild(badge);
            historyItem.appendChild(confidenceBadge);
            historyItem.appendChild(timestamp);

            historyList.appendChild(historyItem);
        });
    }
});