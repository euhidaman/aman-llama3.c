/* Inference for Llama-3 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
#include "win.h"
#else
#include <unistd.h>
#include <sys/mman.h>
#endif

#define VOCAB_SIZE 512
// ----------------------------------------------------------------------------
// Transformer model configuration based on params.py

typedef struct
{
    int dim;        // transformer dimension (default: 128)
    int hidden_dim; // for ffn layers (computed from dim)
    int n_layers;   // number of layers (default: 12)
    int n_heads;    // number of query heads (default: 4)
    int n_kv_heads; // number of key/value heads (same as n_heads)
    int vocab_size; // vocabulary size (512 from tokenizer)
    int seq_len;    // max sequence length (default: 512)
    float norm_eps; // normalization epsilon (default: 1e-5)
} Config;

typedef struct
{
    // token embedding table
    float *token_embedding_table; // (vocab_size, dim)
    // weights for rmsnorms
    float *rms_att_weight; // (layer, dim) rmsnorm weights
    float *rms_ffn_weight; // (layer, dim)
    // weights for matmuls
    float *wq; // (layer, dim, n_heads * head_size)
    float *wk; // (layer, dim, n_kv_heads * head_size)
    float *wv; // (layer, dim, n_kv_heads * head_size)
    float *wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float *w1; // (layer, hidden_dim, dim)
    float *w2; // (layer, dim, hidden_dim)
    float *w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float *rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits
    float *wcls;
} TransformerWeights;

typedef struct
{
    // current wave of activations
    float *x;      // activation at current time stamp (dim,)
    float *xb;     // same, but inside a residual branch (dim,)
    float *xb2;    // additional buffer for convenience (dim,)
    float *hb;     // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2;    // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q;      // query (dim,)
    float *k;      // key (dim,)
    float *v;      // value (dim,)
    float *att;    // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float *key_cache;   // (layer, seq_len, dim)
    float *value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct
{
    Config config;
    TransformerWeights weights;
    RunState state;
    int fd;
    float *data;
    ssize_t file_size;
} Transformer;

void malloc_run_state(RunState *s, Config *p)
{
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));

    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->key_cache || !s->value_cache || !s->att || !s->logits)
    {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState *s)
{
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config *p, float *ptr, int shared_weights)
{
    int head_size = p->dim / p->n_heads;
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip RoPE freq_cis_real
    ptr += p->seq_len * head_size / 2; // skip RoPE freq_cis_imag
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(char *checkpoint, Config *config, TransformerWeights *weights,
                     int *fd, float **data, ssize_t *file_size)
{
    FILE *file = fopen(checkpoint, "rb");
    if (!file)
    {
        fprintf(stderr, "Couldn't open file %s\n", checkpoint);
        exit(EXIT_FAILURE);
    }

    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1)
    {
        exit(EXIT_FAILURE);
    }
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);

    // figure out the file size
    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    fclose(file);

    // memory map the weights
    *fd = open(checkpoint, O_RDONLY);
    if (*fd == -1)
    {
        fprintf(stderr, "open failed!\n");
        exit(EXIT_FAILURE);
    }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED)
    {
        fprintf(stderr, "mmap failed!\n");
        exit(EXIT_FAILURE);
    }
    float *weights_ptr = *data + sizeof(Config) / sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, char *checkpoint_path)
{
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer *t)
{
    if (t->data != MAP_FAILED)
    {
        munmap(t->data, t->file_size);
    }
    if (t->fd != -1)
    {
        close(t->fd);
    }
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks

void rmsnorm(float *o, float *x, float *weight, int size)
{
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
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float *x, int size)
{
    float max_val = x[0];
    for (int i = 1; i < size; i++)
    {
        if (x[i] > max_val)
        {
            max_val = x[i];
        }
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

void matmul(float *xout, float *x, float *w, int n, int d)
{
    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < d; i++)
    {
        float val = 0.0f;
        for (int j = 0; j < n; j++)
        {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

float *forward(Transformer *transformer, int token, int pos)
{
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;
    RunState *s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int head_size = dim / p->n_heads;

    float *content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(*x));

    for (unsigned long long l = 0; l < p->n_layers; l++)
    {
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        int loff = l * p->seq_len * kv_dim;
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

        for (int i = 0; i < dim; i += 2)
        {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++)
            {
                float *vec = v == 0 ? s->q : s->k;
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        int h;
#pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++)
        {
            float *q = s->q + h * head_size;
            float *att = s->att + h * p->seq_len;

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

            softmax(att, pos + 1);

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

        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

        for (int i = 0; i < dim; i++)
        {
            x[i] += s->xb2[i];
        }

        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        matmul(s->hb, s->xb, w->w1 + l * dim * p->hidden_dim, dim, p->hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * p->hidden_dim, dim, p->hidden_dim);

        for (int i = 0; i < p->hidden_dim; i++)
        {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        matmul(s->xb, s->hb, w->w2 + l * dim * p->hidden_dim, p->hidden_dim, dim);

        for (int i = 0; i < dim; i++)
        {
            x[i] += s->xb[i];
        }
    }

    rmsnorm(x, x, w->rms_final_weight, dim);
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// Tokenizer implementation

typedef struct
{
    char *str;
    int id;
} TokenIndex;

typedef struct
{
    char **vocab;
    float *vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores single-byte strings (based on your vocab size of 512)
} Tokenizer;

int compare_tokens(const void *a, const void *b)
{
    return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

void build_tokenizer(Tokenizer *t, char *tokenizer_path, int vocab_size)
{
    printf("Model config vocab_size: %d\n", vocab_size);
    if (vocab_size <= 0 || vocab_size > 512)
    {
        fprintf(stderr, "Error: Vocabulary size must be between 1 and 512\n");
        exit(EXIT_FAILURE);
    }

    t->vocab_size = vocab_size;
    printf("Initializing tokenizer with vocab_size: %d\n", vocab_size); // Debug print

    t->vocab = (char **)malloc(vocab_size * sizeof(char *));
    t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;

    // Initialize byte_pieces (for handling byte-level tokens)
    for (int i = 0; i < 256; i++)
    {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    FILE *file = fopen(tokenizer_path, "rb");
    if (!file)
    {
        fprintf(stderr, "Couldn't open tokenizer file: %s\n", tokenizer_path);
        exit(EXIT_FAILURE);
    }
    printf("Tokenizer file opened successfully.\n");

    // Get file size for verification
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    printf("Tokenizer file size: %ld bytes\n", file_size);

    // Expected minimum file size calculation
    long expected_min_size = sizeof(int) +                  // max_token_length
                             (vocab_size * sizeof(float)) + // vocab_scores
                             vocab_size;                    // minimum 1 byte per token string
    printf("Expected minimum file size: %ld bytes\n", expected_min_size);

    if (file_size < expected_min_size)
    {
        fprintf(stderr, "Error: Tokenizer file is too small. Expected at least %ld bytes, got %ld bytes\n",
                expected_min_size, file_size);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // Read max_token_length
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1)
    {
        fprintf(stderr, "Failed to read max_token_length from tokenizer file.\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    printf("Max token length: %d\n", t->max_token_length);

    // Read vocabulary scores and tokens with additional checks
    int len;
    long current_pos;
    for (int i = 0; i < vocab_size; i++)
    {
        current_pos = ftell(file);
        if (current_pos >= file_size)
        {
            fprintf(stderr, "Unexpected end of file at position %ld while reading token %d\n",
                    current_pos, i);
            fclose(file);
            exit(EXIT_FAILURE);
        }

        if (fread(&t->vocab_scores[i], sizeof(float), 1, file) != 1)
        {
            fprintf(stderr, "Failed to read vocab_scores[%d] from tokenizer file at position %ld\n",
                    i, current_pos);
            fclose(file);
            exit(EXIT_FAILURE);
        }

        current_pos = ftell(file);
        if (fread(&len, sizeof(int), 1, file) != 1)
        {
            fprintf(stderr, "Failed to read token length for vocab[%d] from tokenizer file at position %ld\n",
                    i, current_pos);
            fclose(file);
            exit(EXIT_FAILURE);
        }

        // Sanity check for token length
        if (len <= 0 || len > t->max_token_length)
        {
            fprintf(stderr, "Invalid token length %d for vocab[%d]. Must be between 1 and %d\n",
                    len, i, t->max_token_length);
            fclose(file);
            exit(EXIT_FAILURE);
        }

        t->vocab[i] = (char *)malloc(len + 1);
        if (!t->vocab[i])
        {
            fprintf(stderr, "malloc failed for vocab[%d]\n", i);
            fclose(file);
            exit(EXIT_FAILURE);
        }

        current_pos = ftell(file);
        if (fread(t->vocab[i], len, 1, file) != 1)
        {
            fprintf(stderr, "Failed to read token string for vocab[%d] at position %ld\n",
                    i, current_pos);
            fclose(file);
            exit(EXIT_FAILURE);
        }
        t->vocab[i][len] = '\0';
        printf("Loaded token %d: %s (score: %f)\n", i, t->vocab[i], t->vocab_scores[i]);
    }

    fclose(file);
    printf("Tokenizer built successfully.\n");
}

void free_tokenizer(Tokenizer *t)
{
    for (int i = 0; i < t->vocab_size; i++)
    {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char *decode(Tokenizer *t, int prev_token, int token)
{
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ')
    {
        piece++;
    }
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1)
    {
        piece = (char *)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece)
{
    if (piece == NULL)
    {
        return;
    }
    if (piece[0] == '\0')
    {
        return;
    }
    if (piece[1] == '\0')
    {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val)))
        {
            return;
        }
    }
    printf("%s", piece);
}

// ----------------------------------------------------------------------------
// Sampling implementation

typedef struct
{
    float prob;
    int index;
} ProbIndex;

typedef struct
{
    int vocab_size;
    ProbIndex *probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

unsigned int random_u32(unsigned long long *state)
{
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

int sample_mult(float *probabilities, int n, float coin)
{
    float cdf = 0.0f;
    for (int i = 0; i < n; i++)
    {
        cdf += probabilities[i];
        if (coin < cdf)
        {
            return i;
        }
    }
    return n - 1;
}

int compare(const void *a, const void *b)
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
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++)
    {
        if (probabilities[i] >= cutoff)
        {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++)
    {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp)
        {
            last_idx = i;
            break;
        }
    }

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

int sample(Sampler *sampler, float *logits)
{
    int next;
    if (sampler->temperature == 0.0f)
    {
        next = sample_argmax(logits, sampler->vocab_size);
    }
    else
    {
        for (int q = 0; q < sampler->vocab_size; q++)
        {
            logits[q] /= sampler->temperature;
        }
        softmax(logits, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1)
        {
            next = sample_mult(logits, sampler->vocab_size, coin);
        }
        else
        {
            next = sample_topp(logits, sampler->vocab_size, sampler->topp,
                               sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// Main entry point

void error_usage()
{
    fprintf(stderr, "Usage:   run <checkpoint> <tokenizer> [options]\n");
    fprintf(stderr, "Example: run model.bin tokenizer.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    exit(EXIT_FAILURE);
}

void encode(Tokenizer *t, const char *text, int bos, int eos, int *tokens, int *n_tokens)
{
    // Start with BOS (beginning of sequence) token if requested
    *n_tokens = 0;
    if (bos)
    {
        tokens[(*n_tokens)++] = 1;
    }

    // Simple space-based tokenization for demonstration
    char *text_copy = strdup(text);
    char *token = strtok(text_copy, " ");

    while (token != NULL && *n_tokens < t->max_token_length)
    {
        // Find the token in vocabulary
        for (int i = 0; i < t->vocab_size; i++)
        {
            if (strcmp(t->vocab[i], token) == 0)
            {
                tokens[(*n_tokens)++] = i;
                break;
            }
        }
        token = strtok(NULL, " ");
    }

    // Add EOS (end of sequence) token if requested
    if (eos)
    {
        tokens[(*n_tokens)++] = 2;
    }

    free(text_copy);
}

int main(int argc, char *argv[])
{
    char *checkpoint_path = NULL;
    char *tokenizer_path = NULL;
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 256;
    char *prompt = NULL;
    unsigned long long rng_seed = 0;

    // Check for minimum number of arguments
    if (argc < 3)
    {
        error_usage();
    }

    // Parse checkpoint and tokenizer paths
    checkpoint_path = argv[1];
    tokenizer_path = argv[2];

    // Parse optional arguments
    for (int i = 3; i < argc; i += 2)
    {
        if (i + 1 >= argc)
        {
            error_usage();
        }
        if (argv[i][0] != '-')
        {
            error_usage();
        }
        if (strlen(argv[i]) != 2)
        {
            error_usage();
        }
        switch (argv[i][1])
        {
        case 't':
            temperature = atof(argv[i + 1]);
            break;
        case 'p':
            topp = atof(argv[i + 1]);
            break;
        case 's':
            rng_seed = atoi(argv[i + 1]);
            break;
        case 'n':
            steps = atoi(argv[i + 1]);
            break;
        case 'i':
            prompt = argv[i + 1];
            break;
        default:
            error_usage();
        }
    }

    // Set default random seed if not provided
    if (rng_seed <= 0)
    {
        rng_seed = (unsigned int)time(NULL);
    }

    // Validate temperature and topp values
    if (temperature < 0.0)
    {
        temperature = 0.0;
    }
    if (topp < 0.0 || topp > 1.0)
    {
        topp = 0.9;
    }

    // Initialize the transformer model
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);

    // Adjust steps if necessary
    if (steps == 0 || steps > transformer.config.seq_len)
    {
        steps = transformer.config.seq_len;
    }

    // Initialize the tokenizer
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // Initialize the sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // Handle empty prompt
    char *empty_prompt = "";
    if (prompt == NULL)
    {
        prompt = empty_prompt;
    }

    // Tokenize the input prompt
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int));
    if (prompt_tokens == NULL)
    {
        fprintf(stderr, "malloc failed for prompt tokens\n");
        exit(EXIT_FAILURE);
    }

    encode(&tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

    if (num_prompt_tokens < 1)
    {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // Generate text
    long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;

    while (pos < steps)
    {
        float *logits = forward(&transformer, token, pos);

        if (pos < num_prompt_tokens - 1)
        {
            next = prompt_tokens[pos + 1];
        }
        else
        {
            next = sample(&sampler, logits);
        }
        pos++;

        if (next == 1)
        {
            break; // End of sequence token
        }

        // Decode and print the generated token
        char *piece = decode(&tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;

        // Track time for tokens per second calculation
        if (start == 0)
        {
            start = time(NULL) * 1000;
        }
    }
    printf("\n");

    // Print tokens per second
    if (pos > 1)
    {
        long end = time(NULL) * 1000;
        fprintf(stderr, "achieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
    }

    // Clean up
    free(prompt_tokens);
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);

    return 0;
}