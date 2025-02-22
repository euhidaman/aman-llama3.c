/* Inference for Llama3 Transformer model in pure C */
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <stdint.h>
#if defined _WIN32
#include "win.h"
#else
#include <unistd.h>
#include <sys/mman.h>
#endif

// ----------------------------------------------------------------------------
// Transformer model

typedef struct
{
    int dim;          // transformer dimension (256)
    int n_layers;     // number of layers (12)
    int n_heads;      // number of attention heads (4)
    int n_kv_heads;   // number of key/value heads (4)
    int vocab_size;   // vocabulary size (512)
    int seq_len;      // max sequence length (512)
    float norm_eps;   // RMSNorm epsilon (1e-5)
    int hidden_dim;   // hidden dimension (1024)
    int multiple_of;  // hidden layer size multiple (256)
    float rope_theta; // RoPE theta value (10000.0)
} Config;

typedef struct
{
    // token embedding table
    float *token_embedding_table; // (vocab_size, dim)
    // RMSNorm weights
    float *rms_att_weight; // (layer, dim)
    float *rms_ffn_weight; // (layer, dim)
    // attention weights
    float *wq; // (layer, dim, n_heads * head_size)
    float *wk; // (layer, dim, n_kv_heads * head_size)
    float *wv; // (layer, dim, n_kv_heads * head_size)
    float *wo; // (layer, n_heads * head_size, dim)
    // feed-forward weights
    float *w1; // (layer, hidden_dim, dim)
    float *w2; // (layer, dim, hidden_dim)
    float *w3; // (layer, hidden_dim, dim)
    // final layer norm
    float *rms_final_weight; // (dim,)
    // (optional) classifier weights
    float *wcls;
} TransformerWeights;

typedef struct
{
    // activation buffers
    float *x;      // current token embedding
    float *xb;     // attention branch buffer
    float *xb2;    // second attention buffer
    float *hb;     // FFN hidden buffer
    float *hb2;    // second FFN hidden buffer
    float *q;      // query vectors
    float *k;      // key vectors
    float *v;      // value vectors
    float *att;    // attention scores
    float *logits; // output logits
    // key-value cache
    float *key_cache;   // (layer, seq_len, dim)
    float *value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct
{
    Config config;              // model configuration
    TransformerWeights weights; // model weights
    RunState state;             // runtime state
    int fd;                     // file descriptor for memory mapping
    float *data;                // memory mapped data pointer
    ssize_t file_size;          // checkpoint file size
} Transformer;

// Add these prototypes near the top of the file, after struct definitions
void malloc_run_state(RunState *s, Config *p);
void free_run_state(RunState *s);
void memory_map_weights(TransformerWeights *w, Config *p, float *ptr);
void read_checkpoint(char *checkpoint, Config *config, TransformerWeights *weights,
                     int *fd, float **data, ssize_t *file_size);
void build_transformer(Transformer *t, char *checkpoint_path);
void free_transformer(Transformer *t);

void build_transformer(Transformer *t, char *checkpoint_path)
{
    // Read in the Config and the Weights from the checkpoint
    read_checkpoint(
        checkpoint_path,
        &t->config,
        &t->weights,
        &t->fd,
        &t->data,
        &t->file_size);

    // Allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer *t)
{
    // Close the memory mapping
    if (t->data != MAP_FAILED)
    {
        munmap(t->data, t->file_size);
    }
    if (t->fd != -1)
    {
        close(t->fd);
    }

    // Free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// Memory management and initialization

void malloc_run_state(RunState *s, Config *p)
{
    // Calculate KV dimensions
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;

    // Calculate buffer sizes
    size_t x_size = p->dim * sizeof(float);
    size_t xb_size = p->dim * sizeof(float);
    size_t xb2_size = p->dim * sizeof(float);
    size_t hb_size = p->hidden_dim * sizeof(float);
    size_t hb2_size = p->hidden_dim * sizeof(float);
    size_t q_size = p->dim * sizeof(float);
    size_t k_size = kv_dim * sizeof(float);
    size_t v_size = kv_dim * sizeof(float);
    size_t att_size = p->n_heads * p->seq_len * sizeof(float);
    size_t logits_size = p->vocab_size * sizeof(float);
    size_t key_cache_size = p->n_layers * p->seq_len * kv_dim * sizeof(float);
    size_t value_cache_size = p->n_layers * p->seq_len * kv_dim * sizeof(float);

    // Print debug information
    fprintf(stderr, "\nRunState allocation details:\n");
    fprintf(stderr, "Model dimensions:\n");
    fprintf(stderr, "- dim: %d\n", p->dim);
    fprintf(stderr, "- kv_dim: %d\n", kv_dim);
    fprintf(stderr, "- hidden_dim: %d\n", p->hidden_dim);
    fprintf(stderr, "- n_layers: %d\n", p->n_layers);
    fprintf(stderr, "- n_heads: %d\n", p->n_heads);
    fprintf(stderr, "- seq_len: %d\n", p->seq_len);

    fprintf(stderr, "\nBuffer sizes:\n");
    fprintf(stderr, "- x:          %8zu bytes\n", x_size);
    fprintf(stderr, "- xb:         %8zu bytes\n", xb_size);
    fprintf(stderr, "- xb2:        %8zu bytes\n", xb2_size);
    fprintf(stderr, "- hb:         %8zu bytes\n", hb_size);
    fprintf(stderr, "- hb2:        %8zu bytes\n", hb2_size);
    fprintf(stderr, "- q:          %8zu bytes\n", q_size);
    fprintf(stderr, "- k:          %8zu bytes\n", k_size);
    fprintf(stderr, "- v:          %8zu bytes\n", v_size);
    fprintf(stderr, "- att:        %8zu bytes\n", att_size);
    fprintf(stderr, "- logits:     %8zu bytes\n", logits_size);
    fprintf(stderr, "- key_cache:  %8zu bytes\n", key_cache_size);
    fprintf(stderr, "- value_cache:%8zu bytes\n", value_cache_size);

    size_t total_size = x_size + xb_size + xb2_size + hb_size + hb2_size +
                        q_size + k_size + v_size + att_size + logits_size +
                        key_cache_size + value_cache_size;

    fprintf(stderr, "\nTotal allocation size: %.2f MB\n", total_size / (1024.0 * 1024.0));
    fprintf(stderr, "Attempting allocations...\n");

    // Perform allocations with debug output
    s->x = calloc(p->dim, sizeof(float));
    fprintf(stderr, "- x allocated: %s\n", s->x ? "success" : "FAILED");

    s->xb = calloc(p->dim, sizeof(float));
    fprintf(stderr, "- xb allocated: %s\n", s->xb ? "success" : "FAILED");

    s->xb2 = calloc(p->dim, sizeof(float));
    fprintf(stderr, "- xb2 allocated: %s\n", s->xb2 ? "success" : "FAILED");

    s->hb = calloc(p->hidden_dim, sizeof(float));
    fprintf(stderr, "- hb allocated: %s\n", s->hb ? "success" : "FAILED");

    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    fprintf(stderr, "- hb2 allocated: %s\n", s->hb2 ? "success" : "FAILED");

    s->q = calloc(p->dim, sizeof(float));
    fprintf(stderr, "- q allocated: %s\n", s->q ? "success" : "FAILED");

    s->k = calloc(kv_dim, sizeof(float));
    fprintf(stderr, "- k allocated: %s\n", s->k ? "success" : "FAILED");

    s->v = calloc(kv_dim, sizeof(float));
    fprintf(stderr, "- v allocated: %s\n", s->v ? "success" : "FAILED");

    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    fprintf(stderr, "- att allocated: %s\n", s->att ? "success" : "FAILED");

    s->logits = calloc(p->vocab_size, sizeof(float));
    fprintf(stderr, "- logits allocated: %s\n", s->logits ? "success" : "FAILED");

    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    fprintf(stderr, "- key_cache allocated: %s\n", s->key_cache ? "success" : "FAILED");

    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    fprintf(stderr, "- value_cache allocated: %s\n", s->value_cache ? "success" : "FAILED");

    // Check all allocations
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q ||
        !s->k || !s->v || !s->att || !s->logits ||
        !s->key_cache || !s->value_cache)
    {
        fprintf(stderr, "\nERROR: Memory allocation failed!\n");
        fprintf(stderr, "Required memory: %.2f MB\n", total_size / (1024.0 * 1024.0));
        free_run_state(s);
        exit(EXIT_FAILURE);
    }

    fprintf(stderr, "\nAll memory allocations successful!\n");
}

void free_run_state(RunState *s)
{
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config *p, float *ptr)
{
    // Calculate dimensions
    int head_size = p->dim / p->n_heads;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    size_t running_offset = 0;

    fprintf(stderr, "\nDebug: Memory mapping weights\n");
    fprintf(stderr, "head_size: %d\n", head_size);
    fprintf(stderr, "kv_dim: %d\n", kv_dim);

    // Map token embeddings
    w->token_embedding_table = ptr;
    running_offset = p->vocab_size * p->dim;
    ptr += running_offset;
    fprintf(stderr, "token_embedding offset: %zu bytes\n", running_offset * sizeof(float));

    // Map attention weights for each layer
    size_t layer_offset = 0;
    for (int l = 0; l < p->n_layers; l++)
    {
        // RMSNorm weights
        if (l == 0)
            w->rms_att_weight = ptr;
        ptr += p->dim;
        layer_offset = p->dim;
        running_offset += layer_offset;

        fprintf(stderr, "Layer %d rms_att offset: %zu bytes\n", l, layer_offset * sizeof(float));
    }

    // Query weights
    layer_offset = p->dim * p->dim;
    for (int l = 0; l < p->n_layers; l++)
    {
        if (l == 0)
            w->wq = ptr;
        ptr += layer_offset;
        running_offset += layer_offset;
        fprintf(stderr, "Layer %d wq offset: %zu bytes\n", l, layer_offset * sizeof(float));
    }

    // Key weights
    layer_offset = p->dim * kv_dim;
    for (int l = 0; l < p->n_layers; l++)
    {
        if (l == 0)
            w->wk = ptr;
        ptr += layer_offset;
        running_offset += layer_offset;
        fprintf(stderr, "Layer %d wk offset: %zu bytes\n", l, layer_offset * sizeof(float));
    }

    // Value weights
    for (int l = 0; l < p->n_layers; l++)
    {
        if (l == 0)
            w->wv = ptr;
        ptr += layer_offset;
        running_offset += layer_offset;
        fprintf(stderr, "Layer %d wv offset: %zu bytes\n", l, layer_offset * sizeof(float));
    }

    // Output projection
    layer_offset = p->dim * p->dim;
    for (int l = 0; l < p->n_layers; l++)
    {
        if (l == 0)
            w->wo = ptr;
        ptr += layer_offset;
        running_offset += layer_offset;
        fprintf(stderr, "Layer %d wo offset: %zu bytes\n", l, layer_offset * sizeof(float));
    }

    // FFN weights
    for (int l = 0; l < p->n_layers; l++)
    {
        if (l == 0)
            w->rms_ffn_weight = ptr;
        ptr += p->dim;
        layer_offset = p->dim;
        running_offset += layer_offset;
        fprintf(stderr, "Layer %d rms_ffn offset: %zu bytes\n", l, layer_offset * sizeof(float));
    }

    // FFN projections
    layer_offset = p->dim * p->hidden_dim;
    for (int l = 0; l < p->n_layers; l++)
    {
        if (l == 0)
            w->w1 = ptr;
        ptr += layer_offset;
        running_offset += layer_offset;
        fprintf(stderr, "Layer %d w1 offset: %zu bytes\n", l, layer_offset * sizeof(float));
    }

    layer_offset = p->hidden_dim * p->dim;
    for (int l = 0; l < p->n_layers; l++)
    {
        if (l == 0)
            w->w2 = ptr;
        ptr += layer_offset;
        running_offset += layer_offset;
        fprintf(stderr, "Layer %d w2 offset: %zu bytes\n", l, layer_offset * sizeof(float));
    }

    layer_offset = p->dim * p->hidden_dim;
    for (int l = 0; l < p->n_layers; l++)
    {
        if (l == 0)
            w->w3 = ptr;
        ptr += layer_offset;
        running_offset += layer_offset;
        fprintf(stderr, "Layer %d w3 offset: %zu bytes\n", l, layer_offset * sizeof(float));
    }

    // Final norm
    w->rms_final_weight = ptr;
    ptr += p->dim;
    running_offset += p->dim;

    // Share classifier weights
    w->wcls = w->token_embedding_table;

    fprintf(stderr, "Total weights mapped: %zu bytes\n", running_offset * sizeof(float));
}

void read_checkpoint(char *checkpoint, Config *config, TransformerWeights *weights,
                     int *fd, float **data, ssize_t *file_size)
{

    // Open file first to get size
    FILE *file = fopen(checkpoint, "rb");
    if (!file)
    {
        fprintf(stderr, "Error: Couldn't open file %s\n", checkpoint);
        exit(EXIT_FAILURE);
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Read config header (40 bytes)
    if (fread(config, sizeof(Config), 1, file) != 1)
    {
        fprintf(stderr, "Error: Failed to read config\n");
        exit(EXIT_FAILURE);
    }

    // Print config for debugging
    fprintf(stderr, "\nModel Configuration:\n");
    fprintf(stderr, "- Dimension: %d\n", config->dim);
    fprintf(stderr, "- Layers: %d\n", config->n_layers);
    fprintf(stderr, "- Heads: %d\n", config->n_heads);
    fprintf(stderr, "- KV Heads: %d\n", config->n_kv_heads);
    fprintf(stderr, "- Vocab Size: %d\n", config->vocab_size);
    fprintf(stderr, "- Sequence Length: %d\n", config->seq_len);
    fprintf(stderr, "- Hidden Dim: %d\n", config->hidden_dim);

    fclose(file);

    // Memory map the file
    *fd = open(checkpoint, O_RDONLY);
    if (*fd == -1)
    {
        fprintf(stderr, "Error: Failed to open for memory mapping\n");
        exit(EXIT_FAILURE);
    }

    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED)
    {
        fprintf(stderr, "Error: Memory mapping failed\n");
        close(*fd);
        exit(EXIT_FAILURE);
    }

    // Skip config header for weight mapping
    float *weights_ptr = *data + 10; // 40 bytes = 10 floats
    memory_map_weights(weights, config, weights_ptr);
}

// ----------------------------------------------------------------------------
// Neural net operations

void rmsnorm(float *output, float *x, float *weight, int size)
{
    // RMSNorm: x = x / sqrt(mean(x^2)) * weight
    float ss = 0.0f;
    for (int j = 0; j < size; j++)
    {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);

    for (int j = 0; j < size; j++)
    {
        output[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float *x, int size)
{
    float max_val = x[0];
    for (int i = 1; i < size; i++)
    {
        if (x[i] > max_val)
            max_val = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    for (int i = 0; i < size; i++)
    {
        x[i] /= sum;
    }
}

void matmul(float *out, float *x, float *w, int n, int d)
{
    // W (d,n) @ x (n,) -> out (d,)
    int i;
#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
    for (i = 0; i < d; i++)
    {
        float val = 0.0f;
        for (int j = 0; j < n; j++)
        {
            val += w[i * n + j] * x[j];
        }
        out[i] = val;
    }
}

// Applies rotary position embeddings (RoPE) to q and k
void apply_rope(float *q, float *k, int pos, int dim, int head_size, float rope_theta)
{
    for (int i = 0; i < dim; i += 2)
    {
        int head_dim = i % head_size;
        float freq = 1.0f / powf(rope_theta, head_dim / (float)head_size);
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);

        // Rotate q
        float q0 = q[i];
        float q1 = q[i + 1];
        q[i] = q0 * fcr - q1 * fci;
        q[i + 1] = q0 * fci + q1 * fcr;

        // Rotate k
        float k0 = k[i];
        float k1 = k[i + 1];
        k[i] = k0 * fcr - k1 * fci;
        k[i + 1] = k0 * fci + k1 * fcr;
    }
}

float *forward(Transformer *transformer, int token, int pos)
{
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;
    RunState *s = &transformer->state;

    fprintf(stderr, "\nDebug: Forward pass for token %d at pos %d\n", token, pos);

    // Validate parameters
    if (token >= p->vocab_size)
    {
        fprintf(stderr, "Error: token %d exceeds vocab_size %d\n", token, p->vocab_size);
        exit(1);
    }
    if (pos >= p->seq_len)
    {
        fprintf(stderr, "Error: position %d exceeds seq_len %d\n", pos, p->seq_len);
        exit(1);
    }

    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int head_size = dim / p->n_heads;

    // Copy token embedding
    float *content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(float));

    // Forward all layers
    for (int l = 0; l < p->n_layers; l++)
    {
        // Attention RMSNorm
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        // QKV projections
        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

        // Apply RoPE rotation
        apply_rope(s->q, s->k, pos, dim, head_size, p->rope_theta);

        // Cache k,v at this position
        int loff = l * p->seq_len * kv_dim;
        float *key_cache_row = s->key_cache + loff + pos * kv_dim;
        float *value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(float));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(float));

        // Multihead attention
        int h;
#ifdef _OPENMP
#pragma omp parallel for private(h)
#endif
        for (h = 0; h < p->n_heads; h++)
        {
            float *q = s->q + h * head_size;
            float *att = s->att + h * p->seq_len;

            // Get attention scores for this head
            for (int t = 0; t <= pos; t++)
            {
                float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++)
                {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }

            // Softmax
            softmax(att, pos + 1);

            // Weighted sum of values
            float *xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++)
            {
                float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++)
                {
                    xb[i] += a * v[i];
                }
            }
        }

        // Final projection and residual
        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);
        for (int i = 0; i < dim; i++)
        {
            x[i] += s->xb2[i];
        }

        // FFN RMSNorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        // FFN
        matmul(s->hb, s->xb, w->w1 + l * dim * p->hidden_dim, dim, p->hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * p->hidden_dim, dim, p->hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < p->hidden_dim; i++)
        {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val))); // silu
            val *= s->hb2[i];                    // multiply with w3 output
            s->hb[i] = val;
        }

        // Final projection and residual
        matmul(s->xb, s->hb, w->w2 + l * p->hidden_dim * dim, p->hidden_dim, dim);
        for (int i = 0; i < dim; i++)
        {
            x[i] += s->xb[i];
        }
    }

    // Final RMSNorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // Classifier into logits
    matmul(s->logits, x, w->wcls, dim, p->vocab_size);

    // Add validation before returning
    for (int i = 0; i < p->vocab_size; i++)
    {
        if (!isfinite(s->logits[i]))
        {
            fprintf(stderr, "Warning: Invalid logit value at index %d\n", i);
            s->logits[i] = 0.0f;
        }
    }

    return s->logits;
}

// ----------------------------------------------------------------------------
// Tokenizer

typedef struct
{
    char **vocab;                   // vocabulary strings
    float *vocab_scores;            // vocabulary scores
    int vocab_size;                 // vocabulary size (must be 512)
    int max_token_length;           // max token length in bytes
    unsigned char byte_pieces[512]; // For handling raw bytes
} Tokenizer;

void build_tokenizer(Tokenizer *t, char *tokenizer_path)
{
    // Initialize the tokenizer with fixed vocab size of 512
    t->vocab_size = 512;
    t->vocab = (char **)malloc(t->vocab_size * sizeof(char *));
    t->vocab_scores = (float *)malloc(t->vocab_size * sizeof(float));

    // Initialize byte pieces lookup
    for (int i = 0; i < 256; i++)
    {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    // Read tokenizer from binary file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file)
    {
        fprintf(stderr, "couldn't load %s\n", tokenizer_path);
        exit(EXIT_FAILURE);
    }

    // Read max token length
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1)
    {
        fprintf(stderr, "failed to read max token length\n");
        exit(EXIT_FAILURE);
    }

    // Read each token's score and data
    for (int i = 0; i < t->vocab_size; i++)
    {
        float score;
        size_t len;

        if (fread(&score, sizeof(float), 1, file) != 1 ||
            fread(&len, sizeof(int), 1, file) != 1)
        {
            fprintf(stderr, "failed to read token %d\n", i);
            exit(EXIT_FAILURE);
        }

        t->vocab_scores[i] = score;
        t->vocab[i] = (char *)malloc(len + 1);

        if (fread(t->vocab[i], 1, len, file) != len)
        {
            fprintf(stderr, "failed to read token data %d\n", i);
            exit(EXIT_FAILURE);
        }
        t->vocab[i][len] = '\0';
    }

    fclose(file);
}

void free_tokenizer(Tokenizer *t)
{
    for (int i = 0; i < t->vocab_size; i++)
    {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->vocab_scores);
}

char *decode(Tokenizer *t, int prev_token, int token)
{
    char *piece = t->vocab[token];

    // Handle BOS token special case
    if (prev_token == 1 && piece[0] == ' ')
    {
        piece++;
    }

    // Handle raw byte tokens (e.g., <0x01>)
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1)
    {
        piece = (char *)t->byte_pieces + byte_val * 2;
    }

    return piece;
}

// ----------------------------------------------------------------------------
// Sampling

typedef struct
{
    float prob;
    int index;
} ProbIndex;

typedef struct
{
    int vocab_size;
    ProbIndex *probindex; // Buffer for top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed)
{
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->probindex = malloc(vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *sampler)
{
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state)
{
    // xorshift rng
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state)
{
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_argmax(float *probabilities, int n)
{
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++)
    {
        if (probabilities[i] > max_p)
        {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int compare_probindex(const void *a, const void *b)
{
    ProbIndex *a_ = (ProbIndex *)a;
    ProbIndex *b_ = (ProbIndex *)b;
    if (a_->prob > b_->prob)
        return -1;
    if (a_->prob < b_->prob)
        return 1;
    return 0;
}

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin)
{
    // Sort probabilities in descending order
    for (int i = 0; i < n; i++)
    {
        probindex[i].index = i;
        probindex[i].prob = probabilities[i];
    }
    qsort(probindex, n, sizeof(ProbIndex), compare_probindex);

    // Truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n - 1;
    for (int i = 0; i < n; i++)
    {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp)
        {
            last_idx = i;
            break;
        }
    }

    // Sample from the truncated distribution
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++)
    {
        cdf += probindex[i].prob;
        if (r < cdf)
        {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index;
}

// ----------------------------------------------------------------------------
// Generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps)
{
    char *empty_prompt = "";
    if (prompt == NULL)
    {
        prompt = empty_prompt;
    }

    // Encode the prompt into tokens
    int num_prompt_tokens = 0;
    int max_tokens = strlen(prompt) + 3; // +3 for BOS, EOS, and null terminator
    int *prompt_tokens = (int *)malloc(max_tokens * sizeof(int));

    // Always add BOS (1) token at start
    prompt_tokens[num_prompt_tokens++] = 1; // BOS token

    // Tokenize the prompt
    char *str_buffer = malloc((tokenizer->max_token_length * 2 + 3) * sizeof(char));
    size_t str_len = 0;

    for (char *c = prompt; *c != '\0'; c++)
    {
        // Reset buffer for new UTF-8 character
        if ((*c & 0xC0) != 0x80)
        {
            str_len = 0;
        }

        // Append current byte
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        // Process complete UTF-8 character
        if ((*(c + 1) & 0xC0) != 0x80 || str_len >= 4)
        {
            // Try to find token in vocabulary
            int found = 0;
            for (int i = 0; i < tokenizer->vocab_size; i++)
            {
                if (strcmp(str_buffer, tokenizer->vocab[i]) == 0)
                {
                    prompt_tokens[num_prompt_tokens++] = i;
                    found = 1;
                    break;
                }
            }

            // If not found, encode as bytes
            if (!found)
            {
                for (size_t i = 0; i < str_len; i++)
                {
                    prompt_tokens[num_prompt_tokens++] = (unsigned char)str_buffer[i] + 3;
                }
            }
            str_len = 0;
        }
    }

    free(str_buffer);

    // Generation loop
    long start = 0;               // Timer start
    int next;                     // Next token
    int token = prompt_tokens[0]; // Start with first token
    int pos = 0;                  // Position in sequence
    int prev_token = 0;           // Previous token for decode

    while (pos < steps)
    {
        // Forward the transformer to get logits
        float *logits = forward(transformer, token, pos);

        // Sample next token
        if (pos < num_prompt_tokens - 1)
        {
            // If still processing prompt, use the next prompt token
            next = prompt_tokens[pos + 1];
        }
        else
        {
            // Apply temperature to logits
            if (sampler->temperature != 0.0f)
            {
                for (int q = 0; q < sampler->vocab_size; q++)
                {
                    logits[q] /= sampler->temperature;
                }

                // Apply softmax
                softmax(logits, sampler->vocab_size);

                // Sample using top-p
                float coin = random_f32(&sampler->rng_state);
                if (sampler->topp <= 0 || sampler->topp >= 1)
                {
                    next = sample_argmax(logits, sampler->vocab_size);
                }
                else
                {
                    next = sample_topp(logits, sampler->vocab_size,
                                       sampler->topp, sampler->probindex, coin);
                }
            }
            else
            {
                // Greedy sampling
                next = sample_argmax(logits, sampler->vocab_size);
            }
        }

        pos++;

        // Stop if we encounter EOS (=2) token
        if (next == 2)
            break;

        // Decode and print the token
        char *piece = decode(tokenizer, prev_token, next);
        if (piece != NULL && piece[0] != '\0')
        {
            for (char *p = piece; *p != '\0'; p++)
            {
                if (isprint(*p) || isspace(*p))
                {
                    printf("%c", *p);
                }
            }
            fflush(stdout);
        }

        prev_token = token;
        token = next;

        // Start timer after first iteration
        if (start == 0)
        {
            start = time(NULL);
        }
    }
    printf("\n");

    // Report generation speed
    if (pos > 1)
    {
        long end = time(NULL);
        fprintf(stderr, "achieved tok/s: %f\n", (pos - 1) / (double)(end - start));
    }

    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// Main function and CLI

void error_usage()
{
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>   temperature (0.0 = greedy, 1.0 = original), default 1.0\n");
    fprintf(stderr, "  -p <float>   top-p sampling (0.0 = off, 1.0 = off, 0.9 = recommended), default 0.9\n");
    fprintf(stderr, "  -s <int>     random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>     number of steps to run for, default 256\n");
    fprintf(stderr, "  -i <string>  input prompt\n");
    fprintf(stderr, "  -z <string>  path to tokenizer.bin file\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
    // Default parameters
    char *checkpoint_path = NULL; // model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 0.6f; // Changed from 1.0f to match your settings
    float topp = 0.9f;        // Keep as is
    int steps = 250;          // Changed from 256 to match your max_gen_len
    char *prompt = NULL;
    unsigned long long rng_seed = 0;

    // Parse command line arguments
    if (argc >= 2)
    {
        checkpoint_path = argv[1];
    }
    else
    {
        error_usage();
    }
    for (int i = 2; i < argc; i += 2)
    {
        if (i + 1 >= argc)
            error_usage();
        if (argv[i][0] != '-')
            error_usage();
        if (strlen(argv[i]) != 2)
            error_usage();

        if (argv[i][1] == 't')
            temperature = atof(argv[i + 1]);
        else if (argv[i][1] == 'p')
            topp = atof(argv[i + 1]);
        else if (argv[i][1] == 's')
            rng_seed = atoi(argv[i + 1]);
        else if (argv[i][1] == 'n')
            steps = atoi(argv[i + 1]);
        else if (argv[i][1] == 'i')
            prompt = argv[i + 1];
        else if (argv[i][1] == 'z')
            tokenizer_path = argv[i + 1];
        else
            error_usage();
    }

    // Parameter validation
    if (rng_seed <= 0)
        rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0)
        temperature = 0.0;
    if (topp < 0.0 || topp > 1.0)
        topp = 0.9;
    if (steps < 0)
        steps = 0;

    // Initialize everything
    Transformer transformer;
    Tokenizer tokenizer;
    Sampler sampler;

    // Build the transformer from checkpoint
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len)
    {
        steps = transformer.config.seq_len;
    }

    // Build the tokenizer
    build_tokenizer(&tokenizer, tokenizer_path);

    // Build the sampler
    build_sampler(&sampler, transformer.config.vocab_size,
                  temperature, topp, rng_seed);

    // Run the model
    generate(&transformer, &tokenizer, &sampler, prompt, steps);

    // Cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);

    return 0;
}