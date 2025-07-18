document.addEventListener('DOMContentLoaded', () => {
    console.log('Script loaded');
    const chatBox = document.getElementById('chat-box');
    const theoremInput = document.getElementById('theorem-input');
    const disciplineInput = document.getElementById('discipline-input');
    const domainSelect = document.getElementById('domain-select');
    const refineInput = document.getElementById('refine-input');
    const submitButton = document.getElementById('submit-button');
    const retryButton = document.getElementById('retry-button');
    const thinkingAnimation = document.getElementById('thinking-animation');
    const latexPreview = document.getElementById('latex-preview');
    const latexSuggestions = document.getElementById('latex-suggestions');
    const taskButtons = document.querySelectorAll('.task-btn');
    const styleButtons = document.querySelectorAll('.style-btn');
    const demoMode = document.getElementById('demo-mode');
    const toggleDebug = document.getElementById('toggle-debug');
    const debugPanel = document.getElementById('debug-panel');
    const debugOutput = document.getElementById('debug-output');
    const loadingProgress = document.getElementById('loading-progress');
    const visualizationPanel = document.getElementById('visualization-panel');
    const d3Visualization = document.getElementById('d3-visualization');
    const socket = io();

    let selectedTask = 'prove';
    let selectedStyle = 'unconventional';
    let lastFormData = null;

    // Task buttons
    taskButtons.forEach(button => {
        button.addEventListener('click', () => {
            console.log(`Task selected: ${button.dataset.task}`);
            taskButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            selectedTask = button.dataset.task;
            refineInput.parentElement.classList.toggle('hidden', selectedTask !== 'refine');
            theoremInput.placeholder = selectedTask === 'explore' ? 
                'Enter any mathematical query (e.g., Explore hyperbolic tilings)' : 
                'Enter theorem or query (e.g., \\cosh(c) = \\cosh(a) \\cosh(b))';
        });
    });

    // Style buttons
    styleButtons.forEach(button => {
        button.addEventListener('click', () => {
            console.log(`Style selected: ${button.dataset.style}`);
            styleButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            selectedStyle = button.dataset.style;
        });
    });

    // LaTeX autocompletion
    const latexCommands = [
        { trigger: 'sqrt', suggestion: '\\sqrt{}', example: '\\sqrt{2}' },
        { trigger: 'frac', suggestion: '\\frac{a}{b}', example: '\\frac{1}{2}' },
        { trigger: 'sum', suggestion: '\\sum_{i=0}^{n}', example: '\\sum_{i=0}^{\\infty} \\frac{1}{i^2}' },
        { trigger: 'int', suggestion: '\\int_{a}^{b}', example: '\\int_{0}^{\\pi} \\sin(x) dx' },
        { trigger: 'cosh', suggestion: '\\cosh(x)', example: '\\cosh(a)' },
        { trigger: 'sinh', suggestion: '\\sinh(x)', example: '\\sinh(a)' },
        { trigger: 'parallel', suggestion: '\\parallel', example: 'AB \\parallel CD' }
    ];

    theoremInput.addEventListener('input', () => {
        latexPreview.classList.remove('hidden');
        latexPreview.textContent = theoremInput.value;
        MathJax.typesetPromise([latexPreview]).catch(err => console.error('MathJax error:', err));

        // Autocomplete suggestions
        const value = theoremInput.value.toLowerCase();
        latexSuggestions.innerHTML = '';
        latexSuggestions.classList.add('hidden');
        const matches = latexCommands.filter(cmd => value.includes(cmd.trigger));
        if (matches.length > 0) {
            latexSuggestions.classList.remove('hidden');
            matches.forEach(cmd => {
                const suggestionDiv = document.createElement('div');
                suggestionDiv.className = 'p-2 text-sm text-gray-300 hover:bg-gray-700 cursor-pointer';
                suggestionDiv.innerHTML = `<span>${cmd.suggestion}</span> <span class="text-gray-500">e.g., ${cmd.example}</span>`;
                suggestionDiv.addEventListener('click', () => {
                    theoremInput.value = theoremInput.value.replace(new RegExp(cmd.trigger + '$'), cmd.suggestion);
                    latexPreview.textContent = theoremInput.value;
                    MathJax.typesetPromise([latexPreview]);
                    latexSuggestions.classList.add('hidden');
                });
                latexSuggestions.appendChild(suggestionDiv);
            });
        }
    });

    // Debug toggle
    toggleDebug.addEventListener('click', () => {
        debugPanel.classList.toggle('hidden');
        console.log('Debug panel toggled');
    });

    // Loading bar animation
    function updateLoadingBar() {
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 8 + 2;
            if (progress >= 100) progress = 100;
            loadingProgress.style.width = `${progress}%`;
            if (!thinkingAnimation.classList.contains('hidden') && progress === 100) {
                clearInterval(interval);
            }
        }, 250);
        return interval;
    }

    // Hyperbolic geometry visualization (Poincaré disk)
    function renderHyperbolicVisualization(data, domain) {
        if (domain.includes('hyperbolic') || domain.includes('non_euclidean')) {
            visualizationPanel.classList.remove('hidden');
            d3.select(d3Visualization).selectAll('*').remove();
            const width = 400, height = 400, radius = 180;
            const svg = d3.select(d3Visualization)
                .append('svg')
                .attr('width', '100%')
                .attr('height', height)
                .append('g')
                .attr('transform', `translate(${width/2}, ${height/2})`);

            // Disk boundary
            svg.append('circle')
                .attr('r', radius)
                .attr('fill', 'none')
                .attr('stroke', '#3b82f6')
                .attr('stroke-width', 2);

            // Sample hyperbolic lines (arcs)
            const points = [
                { x: 0, y: 0 },
                { x: 100, y: 100 },
                { x: -100, y: -100 }
            ];
            points.forEach((p, i) => {
                if (i < points.length - 1) {
                    svg.append('path')
                        .attr('d', `M ${p.x},${p.y} A ${radius} ${radius} 0 0 1 ${points[i+1].x},${points[i+1].y}`)
                        .attr('stroke', '#60a5fa')
                        .attr('stroke-width', 2)
                        .attr('fill', 'none');
                }
            });
        }
    }

    // Submit task
    async function submitTask() {
        console.log('Submit clicked');
        const task = selectedTask;
        const domain = domainSelect.value;
        const style = selectedStyle;
        const problem = theoremInput.value.trim();
        const discipline = disciplineInput.value.trim();

        // Validate input
        if (task !== 'conjecture' && task !== 'refine' && task !== 'explore' && !problem) {
            showError('Please enter a theorem or query.');
            return;
        }
        if (task === 'refine' && !refineInput.files.length) {
            showError('Please upload a file for refinement.');
            return;
        }

        // Show user message
        const userMessage = document.createElement('div');
        userMessage.className = 'user-message flex flex-col';
        userMessage.innerHTML = `
            <span class="font-semibold">${task.charAt(0).toUpperCase() + task.slice(1)} in ${domain.replace('_', ' ')} (${style})</span>
            <span class="text-sm text-gray-400">${problem || discipline || 'No specific input'}</span>
        `;
        chatBox.appendChild(userMessage);

        // Show thinking animation and loading bar
        thinkingAnimation.classList.remove('hidden');
        submitButton.classList.add('processing');
        submitButton.disabled = true;
        retryButton.classList.add('hidden');
        const loadingInterval = updateLoadingBar();

        // Prepare form data
        const formData = new FormData();
        formData.append('task', task);
        formData.append('domain', domain);
        formData.append('style', style);
        formData.append('problem', problem);
        formData.append('discipline', discipline);
        if (task === 'conjecture') {
            formData.append('domain2', 'number_theory');
        }
        if (task === 'refine' && refineInput.files.length) {
            formData.append('refine_file', refineInput.files[0]);
        }
        lastFormData = formData;

        try {
            const response = await fetch('/process', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            console.log('Server response:', result);

            clearInterval(loadingInterval);
            thinkingAnimation.classList.add('hidden');
            submitButton.classList.remove('processing');
            submitButton.disabled = false;
            loadingProgress.style.width = '0%';

            if (result.error) {
                throw new Error(result.error);
            }

            // Show bot message with collapsible details
            const output = result.output || 'Server returned empty output.';
            const botMessage = document.createElement('div');
            botMessage.className = 'bot-message flex flex-col';
            botMessage.innerHTML = `
                <button class="toggle-details text-left font-semibold text-blue-400">Toggle Response Details</button>
                <div class="details hidden"><span>${output}</span></div>
            `;
            chatBox.appendChild(botMessage);
            chatBox.scrollTop = chatBox.scrollHeight;

            // Toggle details
            botMessage.querySelector('.toggle-details').addEventListener('click', () => {
                const details = botMessage.querySelector('.details');
                details.classList.toggle('hidden');
            });

            // Show retry button if output is empty
            if (output === 'No output generated by the model.' || output === 'Server returned empty output.') {
                retryButton.classList.remove('hidden');
            }

            // Update debug panel
            debugOutput.textContent = JSON.stringify(result, null, 2);

            // Render visualization
            renderHyperbolicVisualization(result, domain);

            // Re-render MathJax
            MathJax.typesetPromise().catch(err => console.error('MathJax error:', err));
        } catch (error) {
            clearInterval(loadingInterval);
            thinkingAnimation.classList.add('hidden');
            submitButton.classList.remove('processing');
            submitButton.disabled = false;
            loadingProgress.style.width = '0%';
            showError(`Error: ${error.message}`);
            debugOutput.textContent = `Error: ${error.message}`;
            retryButton.classList.remove('hidden');
        }
    }

    // Error message
    function showError(message) {
        const error = document.createElement('div');
        error.className = 'bot-message';
        error.innerHTML = `<span class="text-red-500">${message}</span>`;
        chatBox.appendChild(error);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Submit button
    submitButton.addEventListener('click', submitTask);

    // Retry button
    retryButton.addEventListener('click', async () => {
        console.log('Retry clicked');
        if (lastFormData) {
            await submitTask();
        }
    });

    // Demo mode
    demoMode.addEventListener('click', async () => {
        console.log('Demo mode started');
        theoremInput.value = 'Prove the hyperbolic Pythagorean theorem: \\cosh(c) = \\cosh(a) \\cosh(b).';
        domainSelect.value = 'hyperbolic_geometry';
        selectedTask = 'prove';
        selectedStyle = 'geometric';
        taskButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.task === 'prove'));
        styleButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.style === 'geometric'));
        await submitTask();
    });

    // SocketIO
    socket.on('new_message', data => {
        console.log('SocketIO message received:', data);
        const botMessage = document.createElement('div');
        botMessage.className = 'bot-message flex flex-col';
        botMessage.innerHTML = `
            <button class="toggle-details text-left font-semibold text-blue-400">Toggle Response Details</button>
            <div class="details hidden"><span>${data.output || 'No socket output'}</span></div>
        `;
        chatBox.appendChild(botMessage);
        botMessage.querySelector('.toggle-details').addEventListener('click', () => {
            botMessage.querySelector('.details').classList.toggle('hidden');
        });
        chatBox.scrollTop = chatBox.scrollHeight;
        MathJax.typesetPromise().catch(err => console.error('MathJax error:', err));
        renderHyperbolicVisualization(data, data.domain);
    });

    // Initialize
    taskButtons[0].classList.add('active');
    styleButtons[0].classList.add('active');
    console.log('Initialization complete');
});