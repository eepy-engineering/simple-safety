from nicegui import ui, run 
from SafetyClassifierForward import SafetyClassifierForward
from asyncio import to_thread

# Async function to load the model
model_instance = None
async def load_model():
    global model_instance
    model_instance = await to_thread(
        lambda: SafetyClassifierForward('Qwen/Qwen3-0.6B')
    )


@ui.page('/')
async def page(): 
    global model_instance
    if model_instance is None: 
        loading_dialog = ui.dialog()
        with loading_dialog, ui.column().classes('items-center justify-center h-screen bg-white bg-opacity-95'):
            ui.spinner(size='lg')
            ui.label('Loading safety model...').classes('text-lg text-gray-700')

            # Ininitialize model instance
            loading_dialog.open()
            await load_model()
            loading_dialog.close()

    with ui.column().classes('q-pa-xl'):
        ui.markdown('## **Simple Safety Classifier**')
        ui.markdown(
            """
            This demo runs a lightweight LLM for content moderation.

            Given a text input, it predicts whether the context is **safe** or **unsafe**
            based on a finetuned Qwen 0.6B model.

            The model is tiny, so it will make some mistakes. See if you can find any!
            """
        ).classes('text-grey-7')


        text_input = ui.input(label='Text', placeholder='Input query here', 
            on_change=lambda e: result_label.set_text(f'You typed: {e.value}'), 
            validation={'Input too long': lambda value: len(value) < 256}
        ).classes('w-96 text-black-600')

        result_label = ui.label('').classes('text-lg q-mt-md')


        # Submit Button 
        def on_submit(): 
            text = text_input.value or ""
            # prob = answer_function(text)
            prob_unsafe = model_instance.eval_text(text)
            result_label.classes(remove='text-red-600')
            result_label.classes(remove='text-green-600')

            if prob_unsafe > 0.5: 
                result_label.set_text(
                    f'Your input is unsafe with probability {prob_unsafe:.3f}.'
                )
                result_label.classes(add='text-red-600')

            else: 
                prob_safe = 1.0 - prob_unsafe
                result_label.set_text(
                    f'Your input is safe with probability {prob_safe:.3f}.'
                )
                result_label.classes(add='text-green-600')
        ui.button('Evaluate', on_click=on_submit).classes('q-mt-md')

        ui.markdown(
            """
            Made by Akhil Jalan.
            """
        ).classes('text-grey-7')

ui.run(port = 8098, host="0.0.0.0")
