import { GoogleGenerativeAI } from '@google/generative-ai'

const apiKey = process.env.GEMINI_API_KEY || ''
const genAI = new GoogleGenerativeAI(apiKey)

interface OutputFormat {
    [key: string]: string | string[] | OutputFormat
}

/**
 * Generates structured output using the Gemini API based on given prompts and format.
 * 
 * @param {string} system_prompt - The system prompt to guide the AI's behavior.
 * @param {string|string[]} user_prompt - The user's input prompt(s).
 * @param {OutputFormat} output_format - The desired format for the output.
 * @param {string} default_category - Default category to use if AI can't identify one.
 * @param {boolean} output_value_only - Whether to return only the values of the output.
 * @param {string} chatmodel - The Gemini model to use.
 * @param {number} temperature - The temperature setting for the AI model.
 * @param {number} num_tries - Number of attempts to generate valid output.
 * @param {boolean} verbose - Whether to log detailed information during execution.
 * 
 * @returns {Promise<Array<{question: string, answer: string}>>} An array of question-answer pairs.
 * 
 * @throws {Error} If unable to generate valid output after all attempts.
 */
export async function strict_output(
    system_prompt: string,
    user_prompt: string | string[],
    output_format: OutputFormat,
    default_category: string = '',
    output_value_only: boolean = false,
    chatmodel: string = 'gemini-1.5-pro',
    temperature: number = 1,
    num_tries: number = 3,
    verbose: boolean = false
): Promise<{ question: string; answer: string }[]> {
    console.log('Starting strict output generation...')
    
    const list_input = Array.isArray(user_prompt)
    const dynamic_elements = /<.*?>/.test(JSON.stringify(output_format))
    const list_output = /\[.*?\]/.test(JSON.stringify(output_format))

    for (let i = 0; i < num_tries; i++) {
        console.log('Iteration:', i)
        let output_format_prompt = `\nYou are to output the following in json format: ${JSON.stringify(output_format)}. \nDo not put quotation marks or escape character \\ in the output fields.`

        if (list_output) {
            output_format_prompt += `\nIf output field is a list, classify output into the best element of the list.`
        }
        if (dynamic_elements) {
            output_format_prompt += `\nAny text enclosed by < and > indicates you must generate content to replace it. Example input: Go to <location>, Example output: Go to the garden\nAny output key containing < and > indicates you must generate the key name to replace it. Example input: {'<location>': 'description of location'}, Example output: {school: a place for education}`
        }
        if (list_input) {
            output_format_prompt += `\nGenerate a list of json, one json for each input element.`
        }

        const model = genAI.getGenerativeModel({
            model: chatmodel,
            systemInstruction: system_prompt + output_format_prompt,
        })

        const chat = model.startChat({ history: [] })
        const response = await chat.sendMessage(user_prompt)
        let res = response.response.text()

        if (verbose) {
            console.log('System prompt:', system_prompt + output_format_prompt)
            console.log('\nUser prompt:', user_prompt)
            console.log('\nGemini response:', res)
        }

        try {
            let output: any = JSON.parse(res)
            output = list_input ? output : [output]

            for (let index = 0; index < output.length; index++) {
                for (const key in output_format) {
                    if (/<.*?>/.test(key)) continue
                    if (!(key in output[index])) throw new Error(`${key} not in json output`)

                    if (Array.isArray(output_format[key])) {
                        const choices = output_format[key] as string[]
                        if (Array.isArray(output[index][key])) {
                            output[index][key] = output[index][key][0]
                        }
                        if (!choices.includes(output[index][key]) && default_category) {
                            output[index][key] = default_category
                        }
                        if (output[index][key].includes(':')) {
                            output[index][key] = output[index][key].split(':')[0]
                        }
                    }
                }

                if (output_value_only) {
                    output[index] = Object.values(output[index])
                    if (output[index].length === 1) {
                        output[index] = output[index][0]
                    }
                }
            }

            return list_input ? output : output[0]
        } catch (e) {
            console.log('An exception occurred:', e)
            console.log('Current invalid json format:', res)
        }
    }

    console.log('Finishing strict output generation...')
    return []
}