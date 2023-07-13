# Load necessary packages
library(stm)
library(jiebaR)
library(dplyr)
library(quanteda)
library(quanteda.textstats)
library(quanteda.textmodels)
library(sysfonts)
library(showtext)
showtext_auto(enable = TRUE)
library(furrr)
library(ggplot2)
library(tidyverse)
library(tidytext)
library(stminsights)
library(igraph)
library(ggraph)
library(factoextra)

# Read input data
input_data <- read.csv(file = 'token.csv')

# Drop duplicate thread_id, keeping the longer title
input_data <- input_data[order(input_data$clean, decreasing = TRUE), ]
input_data <- input_data[!duplicated(input_data$thread_id), ]

# Create corpus
corpus <- corpus(input_data, docid_field = "thread_id", text_field = "tokenz") %>%
  tokenizers::tokenize_regex(pattern = " ") %>%
  tokens()

# Add metadata to the corpus
docvars(corpus, "thread_id") <- as.character(input_data$thread_id)
# docvars(corpus, "day") <- as.numeric(input_data$day)

# Create document-feature matrix
dfm <- dfm(corpus)

# Trim the data to remove rare and common words
dfm_trimmed <- dfm_trim(dfm, min_docfreq = 0.004, max_docfreq = 0.99, docfreq_type = "prop")
dfm_trimmed

# Convert the dfm to STM format
stm_dfm <- convert(dfm_trimmed, to = "stm", docvars = docvars(corpus))

# Create stm list
stm <- list(
  documents = stm_dfm$documents,
  vocab = stm_dfm$vocab,
  meta = stm_dfm$meta
)

# Try several different models and compare performance
models <- tibble(K = ((1:10)*5)) %>%
  mutate(
    topic_model = future_map(K, ~stm(
      stm$documents, stm$vocab, K = .,
      data = stm$meta, init.type = "Spectral",
      seed = 2023
    ))
  )

# Evaluate model quality
heldout <- make.heldout(dfm_trimmed)

# Evaluate model quality for each value of K
k_result <- models %>%
  mutate(
    # Calculate exclusivity
    exclusivity = map(topic_model, exclusivity),
    # Calculate semantic coherence
    semantic_coherence = map(topic_model, semanticCoherence, dfm_trimmed),
    # Evaluate held-out likelihood
    eval_heldout = map(topic_model, eval.heldout, heldout$missing),
    # Check residuals
    residual = map(topic_model, checkResiduals, dfm_trimmed),
    # Calculate bound and lfact
    bound = map_dbl(topic_model, function(x) max(x$convergence$bound)),
    lfact = map_dbl(topic_model, function(x) lfactorial(x$settings$dim$K)),
    lbound = bound + lfact,
    iterations = map_dbl(topic_model, function(x) length(x$convergence$bound))
  )

# Create plot to compare model quality for different values of K
k_result %>%
  transmute(
    K,
    `Lower bound` = lbound,
    Residuals = map_dbl(residual, "dispersion"),
    `Semantic coherence` = map_dbl(semantic_coherence, mean),
    `Held-out likelihood` = map_dbl(eval_heldout, "expected.heldout")
  ) %>%
  gather(Metric, Value, -K) %>%
  ggplot(aes(K, Value, color = Metric)) +
  geom_line(size = 1.5, alpha = 0.7, show.legend = FALSE) +
  facet_wrap(~Metric, scales = "free_y") +
  labs(
    x = "K (number of topics)",
    y = NULL,
    title = "Model diagnostics by number of topics",
    subtitle = ""
  )

# Compare exclusivity and semantic coherence for different values of K
k_result %>%
  unnest(c(exclusivity, semantic_coherence)) %>% 
  filter(K %in% ((1:10)*5)) %>%
  group_by(K) %>% 
  summarize(
    exclusivity = mean(exclusivity),
    semantic_coherence = mean(semantic_coherence)
  ) %>% 
  mutate(K = as.factor(K)) %>%
  ggplot(aes(x = semantic_coherence, y = exclusivity, color = K)) +
  geom_point(size = 1, alpha = 0.7, show.legend = FALSE) +
  ggrepel::geom_text_repel(
    aes(label = K),
    size = 5,
    hjust = 0, 
    show.legend = FALSE
  )+
  labs(
    x = "Semantic Coherence (mean)",
    y = "Exclusivity (mean)",
    title = "Comparing exclusivity and semantic coherence",
    subtitle = ""
  )

# Extract the best-performing models for further analysis
model_20 <- models %>% filter(K == 20) %>% pull(topic_model) %>% .[[1]]
model_25 <- models %>% filter(K == 25) %>% pull(topic_model) %>% .[[1]]
model_35 <- models %>% filter(K == 35) %>% pull(topic_model) %>% .[[1]]
model_40 <- models %>% filter(K == 40) %>% pull(topic_model) %>% .[[1]]

# Print the top 10 terms for each topic in each of the best models
T_20 <- labelTopics(model_20, n = 10)
capture.output(T_20, file = "T_20.txt") 
T_25 <- labelTopics(model_25, n = 10)
capture.output(T_25, file = "T_25.txt")
T_35 <- labelTopics(model_35, n = 10)
capture.output(T_35, file = "T_35.txt")
T_40 <- labelTopics(model_40, n = 10)
capture.output(T_40, file = "T_40.txt")

# Find most relevant documents for each topic in each model
findThoughts(model_15, texts = stm[["meta"]][["thread_title"]], n = 4)
findThoughts(model_25, texts = stm[["meta"]][["thread_title"]], n = 4)
findThoughts(model_35, texts = stm[["meta"]][["thread_title"]], n = 4)

# Visualize topic quality for each topic in each model
par(mfrow = c(2,2), mar = c(2, 2, 2, 2))
topicQuality(model_15, documents = stm$documents, main = "model_15")
topicQuality(model_25, documents = stm$documents, main = "model_25")
topicQuality(model_35, documents = stm$documents, main = "model_35")

# Create tables to summarize topic quality for each model
Q_15 <- tibble(
  topic = 1:15,
  exclusivity = exclusivity(model_15),
  semantic_coherence = semanticCoherence(model_15, stm$documents)
) %>% 
  ggplot(aes(semantic_coherence, exclusivity, label = topic)) +
  geom_point() +
  geom_text(nudge_y = .01) +
  theme_bw()

Q_25 <- tibble(
  topic = 1:25,
  exclusivity = exclusivity(model_25),
  semantic_coherence = semanticCoherence(model_25, stm$documents)
) %>% 
  ggplot(aes(semantic_coherence, exclusivity, label = topic)) +
  geom_point() +
  geom_text(nudge_y = .01) +
  theme_bw()

Q_35 <- tibble(
  topic = 1:35,
  exclusivity = exclusivity(model_35),
  semantic_coherence = semanticCoherence(model_35, stm$documents)
) %>% 
  ggplot(aes(semantic_coherence, exclusivity, label = topic)) +
  geom_point() +
  geom_text(nudge_y = .01) +
  theme_bw()

# Write the summary tables to CSV files
write.table(Q_15[["data"]] , file = "Q_15.csv", sep = ",", row.names = FALSE)
write.table(Q_25[["data"]] , file = "Q_25.csv", sep = ",", row.names = FALSE)
write.table(Q_35[["data"]] , file = "Q_35.csv", sep = ",", row.names = FALSE)

dt <- make.dt(model_35, meta=input_data)
write.csv(dt,"35_theta.csv", row.names = FALSE)
