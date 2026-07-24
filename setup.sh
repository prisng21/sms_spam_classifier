mkdir -p ~/.streamlit/

echo "\
[server]\n\
port = $PORT\n\
address = \"0.0.0.0\"\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml
