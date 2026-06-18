import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import Sequential

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Detector de Spam IA", page_icon="🛡️", layout="centered"
)


# 1. Base de dados sintética para treinamento inicial do modelo
@st.cache_resource
def treinar_modelo_spam():
    # Exemplos de Spams (Classe 1) e Não-Spams (Classe 0)
    mensagens_treino = [
        # Spams (1)
        "GANHE DINHEIRO AGORA! Clique no link e mude de vida hoje mesmo!",
        "Você ganhou um prêmio de R$ 50.000! Insira seus dados bancários aqui.",
        "URGENTE: Sua conta foi bloqueada. Regularize seus dados clicando aqui.",
        "Oferta exclusiva de criptomoedas. Lucro garantido em 24 horas!",
        "Parabéns! Seu número foi sorteado para receber um iPhone grátis.",
        "Compre remédios sem receita com descontos inacreditáveis. Acesse já.",
        # Não-Spams / Ham (0)
        "Olá, segue em anexo o relatório técnico da sprint desta semana.",
        "Confirmado nosso almoço de alinhamento amanhã às 12h? Me avisa.",
        "Lembrete: A reunião de revisão do projeto começa em 10 minutos.",
        "Oi mãe, esqueci minha chave em casa, você vai demorar para voltar?",
        "Seu código foi revisado com sucesso. Pode prosseguir com o merge.",
        "Segue o link para o documento compartilhado que você me pediu ontem.",
    ]

    # 1 = Spam, 0 = Seguro
    labels = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int32)

    # Configuração da camada de vetorização de texto
    max_vocab = 1000
    sequence_length = 50

    vectorizer = TextVectorization(
        max_tokens=max_vocab,
        output_mode="int",
        output_sequence_length=sequence_length,
    )
    
    # Adapt também precisa do formato correto de texto
    vectorizer.adapt(np.array(mensagens_treino, dtype=str))

    # Construção do modelo usando a API Sequential do TensorFlow
    model = Sequential(
        [
            vectorizer,
            Embedding(input_dim=max_vocab, output_dim=16, name="embedding"),
            GlobalAveragePooling1D(),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),  # Saída binária (0 a 1)
        ]
    )

    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )

    # CORREÇÃO DEFINITIVA: Convertendo explicitamente para um Tensor de Strings do TensorFlow
    X_treino = tf.constant(mensagens_treino, dtype=tf.string)
    y_treino = tf.constant(labels)

    model.fit(X_treino, y_treino, epochs=30, verbose=0)

    return model


# Inicializa o modelo (usa cache para não treinar a cada clique na tela)
with st.spinner("Inicializando inteligência artificial..."):
    modelo_spam = treinar_modelo_spam()

# --- Interface Gráfica (Streamlit) ---

st.title("🛡️ Classificador de Spam com TensorFlow")
st.write(
    "Copie e cole o conteúdo do e-mail abaixo para verificar se a mensagem é segura ou se trata de um Spam."
)

st.divider()

# Campo de entrada do usuário
email_usuario = st.text_area(
    "Conteúdo do E-mail:",
    placeholder="Cole o texto do e-mail suspeito aqui...",
    height=200,
)

if st.button("Analisar E-mail", use_container_width=True):
    if email_usuario.strip() == "":
        st.warning("Por favor, digite ou cole algum texto para análise.")
    else:
        # CORREÇÃO NA PREDIÇÃO: Passando como um tensor constante de string
        X_pred = tf.constant([email_usuario], dtype=tf.string)
        predicao = modelo_spam.predict(X_pred)[0][0]

        st.subheader("Resultado da Análise:")

        # Threshold padrão de 0.5 para classificação binária
        if predicao >= 0.5:
            confianca = predicao * 100
            st.error("🚨 **Alerta de Spam!**")
            st.write(
                "A inteligência artificial identificou padrões maliciosos ou promocionais agressivos nesta mensagem."
            )
            st.caption(f"Grau de certeza da IA: {confianca:.2f}%")
        else:
            confianca = (1 - predicao) * 100
            st.success("✅ **E-mail Seguro!**")
            st.write(
                "Esta mensagem apresenta características de uma comunicação normal (Ham)."
            )
            st.caption(f"Grau de certeza da IA: {confianca:.2f}%")

st.divider()

# Seção informativa de exemplos exigida no escopo
with st.expander("📌 Visualizar exemplos de treino do Modelo (Dataset)"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Exemplos de SPAM (Sinalizados como 🚨):**")
        st.code(
            "- GANHE DINHEIRO AGORA! link...\n"
            "- Você ganhou um prêmio de R$ 50.000!\n"
            "- URGENTE: Sua conta foi bloqueada.\n"
            "- Oferta exclusiva de criptomoedas.",
            language="text",
        )

    with col2:
        st.markdown("**Exemplos de Mensagens Seguras (Sinalizadas como ✅):**")
        st.code(
            "- Olá, segue em anexo o relatório técnico...\n"
            "- Confirmado nosso almoço de alinhamento amanhã?\n"
            "- Lembrete: A reunião de revisão do projeto...\n"
            "- Seu código foi revisado com sucesso.",
            language="text",
        )
        
