import logging

import streamlit as st

from src.auth import build_auth_service
from src.auth.cognito import AuthError
from src.config import settings
from src.ingestion.bootstrap import build_sample_vectorstore, has_persisted_vectorstore
from src.ui.chat import answer_question
from src.ui.client_ip import resolve_client_ip

logger = logging.getLogger("App")

st.set_page_config(page_title="challenge_epam_LLM", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Preparing the document index...")
def ensure_vectorstore() -> None:
    if not has_persisted_vectorstore() and not build_sample_vectorstore():
        raise RuntimeError("The sample document index could not be created.")


@st.cache_resource(show_spinner="Connecting to the authentication service...")
def get_auth_service():
    return build_auth_service()


def client_ip() -> str:
    return resolve_client_ip(st.context.ip_address, st.context.headers, settings.trusted_proxy_hops)


def clear_session() -> None:
    for key in ("session_id", "access_token", "username"):
        st.session_state.pop(key, None)


def render_auth(auth) -> None:
    st.info("Sign in to ask questions. Each session has a limited token budget.")
    sign_in_tab, sign_up_tab = st.tabs(["Sign in", "Create account"])

    with sign_in_tab:
        with st.form("sign_in"):
            identifier = st.text_input("E-mail or phone (+573001234567)")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in"):
                try:
                    result = auth.login(identifier, password, client_ip())
                except AuthError as error:
                    st.error(str(error))
                except Exception:
                    logger.exception("Sign-in failed.")
                    st.error("Sign-in is unavailable right now. Please try again.")
                else:
                    st.session_state["session_id"] = result.session.session_id
                    st.session_state["access_token"] = result.access_token
                    st.session_state["username"] = result.session.username
                    st.rerun()

    with sign_up_tab:
        with st.form("sign_up"):
            identifier = st.text_input("E-mail or phone (+573001234567)", key="sign_up_identifier")
            password = st.text_input(
                "Password (10+ characters, upper-case, lower-case and a number)",
                type="password",
                key="sign_up_password",
            )
            if st.form_submit_button("Create account"):
                try:
                    result = auth.register(identifier, password, client_ip())
                except AuthError as error:
                    st.error(str(error))
                except Exception:
                    logger.exception("Sign-up failed.")
                    st.error("Sign-up is unavailable right now. Please try again.")
                else:
                    if result.confirmed:
                        st.success("Account created. You can sign in now.")
                    else:
                        st.session_state["pending_confirmation"] = identifier
                        st.success(f"Account created. Enter the code sent to {result.delivery_destination}.")

        pending = st.session_state.get("pending_confirmation")
        if pending:
            with st.form("confirm"):
                code = st.text_input(f"Confirmation code for {pending}")
                if st.form_submit_button("Confirm account"):
                    try:
                        auth.confirm(pending, code)
                    except AuthError as error:
                        st.error(str(error))
                    else:
                        st.session_state.pop("pending_confirmation", None)
                        st.success("Account confirmed. You can sign in now.")


def current_session(auth):
    """Validate the stored session against DynamoDB and Cognito; sign out if invalid."""
    if "session_id" not in st.session_state:
        return None
    try:
        return auth.validate(st.session_state["session_id"], st.session_state["access_token"])
    except AuthError as error:
        clear_session()
        st.warning(str(error))
        return None


ensure_vectorstore()

st.title("challenge_epam_LLM - Colombian & Global Socioeconomic Impact RAG Agent")
st.caption("Strict RAG mode: answers are grounded only in retrieved documents with citations.")

try:
    auth = get_auth_service()
except Exception:
    logger.exception("Authentication service could not be initialised.")
    st.error("The authentication service is unavailable, so questions are disabled.")
    st.stop()

session = current_session(auth)
if session is None:
    render_auth(auth)
    st.stop()

question = st.text_input("Ask a question about your indexed documents")

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving context and generating grounded answer..."):
        result = answer_question(question, auth.token_budget(session))

    st.subheader("Answer")
    st.text(result["answer"])

    st.subheader("Citations")
    if result["citations"]:
        for idx, source in enumerate(result["citations"], start=1):
            st.text(f"Source {idx}: {source}")
    else:
        st.info("No supporting citation available. The assistant abstained.")

    try:  # refresh the counter after the question was charged
        session = auth.sessions.get_active(session.session_id)
    except Exception:
        pass

# Rendered last so the token counter includes the question just asked.
with st.sidebar:
    st.write(f"Signed in as **{session.username}**")
    st.metric("Session tokens left", f"{session.tokens_remaining:,}", help=f"Limit: {session.token_limit:,}")
    if st.button("Sign out"):
        try:
            auth.logout(st.session_state["session_id"], st.session_state["access_token"])
        except Exception:
            logger.exception("Sign-out failed.")
        clear_session()
        st.rerun()
