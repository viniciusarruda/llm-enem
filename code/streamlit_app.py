import os
import traceback
import streamlit as st
from chat_completion_wrapper import OpenAIChatCompletionWrapper, LoadingModelError, DisabledEndpointError, AuthenticationError
from prompt_builder import get_prompt
from logger import logger
from evaluator import get_dataset, get_formated_answer

LOG_STDOUT = True
ENABLE_DOWNLOAD_LOG = True


st.set_page_config(page_title="LLM Enem", layout="wide")


@st.cache_resource
def get_cached_dataset(dataset_name):
    return get_dataset(dataset_name)


def set_dataset():
    st.session_state["dataset"] = get_cached_dataset(st.session_state["prompt_type"])


def show_dataset():
    for question in st.session_state["dataset"][st.session_state["selected_question_area"]].values():
        with st.expander(question["id"], expanded=False):
            st.subheader("Question")
            st.markdown(question["query"].replace("\n", "\n\n"))


def get_input_source_option():
    if st.session_state["input_source"] == "Enem 2022":
        question = st.session_state["dataset"][st.session_state["selected_question_area"]][
            st.session_state["selected_question"]
        ]
    else:
        question = {
            "prompt": get_prompt(
                st.session_state["prompt_type"],
                question_header=st.session_state["question_header"],
                question_statement=st.session_state["question_statement"],
                question_choice_A=st.session_state["question_choice_A"],
                question_choice_B=st.session_state["question_choice_B"],
                question_choice_C=st.session_state["question_choice_C"],
                question_choice_D=st.session_state["question_choice_D"],
                question_choice_E=st.session_state["question_choice_E"],
            )
        }

    return question


def set_llm(openai_api_key):
    if st.session_state["model"] in ["gpt-3.5-turbo", "gpt-4"]:
        try:
            st.session_state["llm"] = OpenAIChatCompletionWrapper(
                model=st.session_state["model"], openai_api_key=openai_api_key
            )
        except AuthenticationError as e:
            st.error(
                "Incorrect API key provided. You can find your API key at https://platform.openai.com/account/api-keys.",
                icon="🚫",
            )
            logger(observation="ERROR", content=traceback.format_exc())
    else:
        raise ValueError(f"Model {st.session_state['model']} is not available.")


def generate():
    question = get_input_source_option()

    with st.spinner("Generating answer..."):
        st.session_state["llm"].new_session()
        answer = st.session_state["llm"](question["prompt"])

    st.divider()
    st.subheader("Answer")
    st.markdown(answer)

    if st.session_state["input_source"] == "Enem 2022":
        pred, gold = get_formated_answer(question, answer)
        logger(observation="After processing LLM output", content=pred)
        if pred == gold:
            logger(observation="Result", content="Answer is correct!")
            st.markdown("✅ Answer is correct!")
        else:
            logger(observation="Result", content="Answer is incorrect!")
            st.markdown("❌ Answer is incorrect!")
            st.markdown(f"The expected answer is:\n\n{gold} {question['choices'][question['gold']]}")

    with st.expander("Data details"):
        st.json(question)


def main():
    if len(st.session_state) == 0:
        st.session_state["dataset"] = None
        st.session_state["llm"] = None

    st.title("LLM Enem")
    st.markdown("Source code: [Github](https://github.com/viniciusarruda/llm-enem)")

    col1, col2 = st.columns(2)

    with col2:
        st.radio("Input source", ("Enem 2022", "Custom"), horizontal=True, key="input_source")

        openai_api_key = st.text_input("OpenAI API KEY", type="password")

        st.selectbox(
            "Model",
            ["gpt-3.5-turbo", "gpt-4"],
            index=0,
            key="model",
            help="Choose the LLM to use to answer the questions.",
            on_change=lambda: set_llm(openai_api_key),
            disabled=len(openai_api_key) == 0,
        )

        if len(openai_api_key) > 0 and st.session_state["llm"] == None:
            set_llm(openai_api_key)

        st.selectbox(
            "Prompt Type",
            ("Zero-shot", "Few-shot", "Few-shot with Chain-of-Thought"),
            key="prompt_type",
            help="The few-shot approach considers three shots.",
            on_change=set_dataset,
        )

        if st.session_state["input_source"] == "Enem 2022":
            if st.session_state["dataset"] == None:
                set_dataset()

            st.selectbox(
                "Area",
                [area for area in st.session_state["dataset"].keys()],
                key="selected_question_area",
            )
            st.selectbox(
                "Select question",
                [
                    question["id"]
                    for question in st.session_state["dataset"][st.session_state["selected_question_area"]].values()
                ],
                key="selected_question",
            )

        if st.button("Answer", disabled=len(openai_api_key) == 0):
            try:
                generate()
            except LoadingModelError:
                st.warning(
                    "The selected model is loading, please wait a moment (~10 min) and try again.",
                    icon="🕑",
                )
                logger(observation="ERROR", content=traceback.format_exc())
            except DisabledEndpointError:
                st.error("The selected model is disabled.", icon="🚫")
                logger(observation="ERROR", content=traceback.format_exc())
            finally:
                log_filepath = logger.save()

            if ENABLE_DOWNLOAD_LOG:
                st.divider()
                with open(log_filepath, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="Download Session Log",
                        data=f,
                        file_name=os.path.basename(log_filepath),
                        mime="text/plain",
                    )

    with col1:
        if st.session_state["input_source"] == "Enem 2022":
            st.header(
                "Enem 2022",
                help="The original Enem exam used to build the ENEM 2022 dataset can be downloaded [here](https://download.inep.gov.br/enem/provas_e_gabaritos/2022_PV_impresso_D1_CD3.pdf) and [here](https://download.inep.gov.br/enem/provas_e_gabaritos/2022_PV_impresso_D2_CD6.pdf).",
            )
            show_dataset()
        else:
            st.header("Custom input")
            st.text_area(
                "Header",
                placeholder="Type here the header of the question.",
                key="question_header",
            )
            st.text_area(
                "Statement",
                placeholder="Type here the statement of the question.",
                key="question_statement",
            )
            n_choices = 5  # st.number_input("Insert the number of choices", value=5, min_value=1)
            for i in range(n_choices):
                letter = chr(ord("A") + i)
                st.text_input(
                    f"{letter}.",
                    placeholder=f"Type here the choice {letter}.",
                    key=f"question_choice_{letter}",
                )


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        logger(observation="ERROR", content=traceback.format_exc())
        logger.save()
        st.error("Internal Error! (if it persists, please contact viniciusferracoarruda@gmail.com)", icon="🚨")
