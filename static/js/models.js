// static/js/models.js
document.addEventListener('DOMContentLoaded', function() {
    // Show/hide hyperparameter tuning options
    const hpTuningCheckbox = document.getElementById('hp_tuning');
    const tuningOptions = document.getElementById('tuning_options');
    const modelTypeInputs = document.querySelectorAll('input[name="model_type"]');
    const tuningDisabledMessage = document.getElementById('tuning_disabled_message');
    
    // Deep learning and ensemble model types that don't support hyperparameter tuning
    const nonTunableModels = ['lstm', 'cnn', 'bert', 'voting', 'stacking'];
    
    if (hpTuningCheckbox) {
        hpTuningCheckbox.addEventListener('change', function() {
            if (this.checked) {
                tuningOptions.style.display = 'block';
            } else {
                tuningOptions.style.display = 'none';
            }
        });
    }
    
    // Handle model type selection
    modelTypeInputs.forEach(input => {
        input.addEventListener('change', function() {
            const selectedModel = this.value;
            
            // Check if hyperparameter tuning should be disabled
            if (nonTunableModels.includes(selectedModel)) {
                if (hpTuningCheckbox) {
                    hpTuningCheckbox.checked = false;
                    hpTuningCheckbox.disabled = true;
                }
                
                if (tuningOptions) {
                    tuningOptions.style.display = 'none';
                }
                
                if (tuningDisabledMessage) {
                    if (['lstm', 'cnn', 'bert'].includes(selectedModel)) {
                        document.getElementById('tuning_disabled_reason').textContent = 
                            'Hyperparameter tuning is not available for deep learning models.';
                    } else {
                        document.getElementById('tuning_disabled_reason').textContent = 
                            'Hyperparameter tuning is not available for ensemble models.';
                    }
                    tuningDisabledMessage.style.display = 'block';
                }
            } else {
                if (hpTuningCheckbox) {
                    hpTuningCheckbox.disabled = false;
                }
                
                if (tuningDisabledMessage) {
                    tuningDisabledMessage.style.display = 'none';
                }
            }
            
            // Show appropriate info section based on model type
            updateModelInfoSection(selectedModel);
        });
    });
    
    // Function to update the model info section
    function updateModelInfoSection(modelType) {
        // Hide all info sections first
        document.querySelectorAll('.model-info-section').forEach(section => {
            section.style.display = 'none';
        });
        
        // Show the relevant section
        let sectionToShow = '';
        
        if (['lstm', 'cnn', 'bert'].includes(modelType)) {
            sectionToShow = 'deep-learning-info';
        } else if (['voting', 'stacking'].includes(modelType)) {
            sectionToShow = 'ensemble-info';
        } else if (['xgboost', 'lightgbm'].includes(modelType)) {
            sectionToShow = 'gradient-boosting-info';
        } else if (['logistic', 'naive_bayes'].includes(modelType)) {
            sectionToShow = 'traditional-info';
        } else if (modelType === 'random_forest') {
            sectionToShow = 'tree-based-info';
        }
        
        if (sectionToShow) {
            const infoSection = document.getElementById(sectionToShow);
            if (infoSection) {
                infoSection.style.display = 'block';
            }
        }
    }
    
    // Initialize the UI based on the default selected model
    const defaultSelectedModel = document.querySelector('input[name="model_type"]:checked');
    if (defaultSelectedModel) {
        updateModelInfoSection(defaultSelectedModel.value);
    }
    
    // Word Embeddings Creation
    const createEmbeddingsBtn = document.getElementById('create-embeddings-btn');
    if (createEmbeddingsBtn) {
        createEmbeddingsBtn.addEventListener('click', function() {
            const spinnerEl = this.querySelector('.spinner-border');
            const textEl = this.querySelector('.btn-text');
            
            // Show spinner, change text
            if (spinnerEl) spinnerEl.style.display = 'inline-block';
            if (textEl) textEl.textContent = 'Creating Embeddings...';
            
            // Disable button
            this.disabled = true;
            
            // Submit the form
            document.getElementById('create-embeddings-form').submit();
        });
    }
});