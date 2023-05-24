# NLP-spell-corrector - Spell Checker with Contextual Error Correction
The repository employs error matrices and incorporates them into a context-sensitive noisy channel model. By leveraging the power of a learned language model, the spell checker can accurately detect and rectify errors, providing comprehensive and reliable suggestions for misspelled or contextually incorrect words.

# Overview
This repository contains an implementation of a spell checker that handles both non-word and real-word errors within a sentential context. The spell checker employs a context-sensitive noisy channel model, utilizing a learned language model and error matrices to provide accurate error detection and correction suggestions.

# Features
* Contextual Error Correction: The spell checker corrects errors based on the most probable correction at the error type-character level, considering the words prior and maximizing the likelihood of obtaining a fully corrected sentence.
* Language Model Integration: By utilizing a language model, the spell checker enhances the accuracy of error correction in a sentential context.
* Noisy Channel Model: The spell checker incorporates a noisy channel model that combines the language model and error matrices to identify and rectify errors effectively.
* Error Types and Prioritization: The spell checker handles at most two errors per word, considering substitution, deletion, and insertion errors. Corrections are prioritized based on the likelihood of each error type.
* Seamless API Integration: The code seamlessly integrates with the provided API, allowing easy initialization, addition of language models and error tables, spell checking, evaluation, and text generation. 

# Example 
<img width="500" alt="image" src="https://github.com/danan08/NLP-spell-corrector/assets/78946759/9f4f473c-c9c8-464d-9f47-c5b8d0529bec">

