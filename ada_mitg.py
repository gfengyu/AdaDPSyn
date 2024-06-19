from data_utils import load_movie_genre
import random
import numpy as np
import re
import argparse
from openai import OpenAI
import math
from scipy.optimize import root_scalar, minimize_scalar

# Using vLLM
# See https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html
openai_api_key = "" # Please add according to your specific set-ups.
openai_api_base = "" # Please add according to your specific set-ups.
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('MIT-G Extraction')
    model_settings = parser.add_argument_group('parameter settings')

    model_settings.add_argument('--num_private_train', type=int, default=80,
                                help='num_private_train=MN')
    model_settings.add_argument('--n_shot', type=int, default=4,
                                help='number of demonstrations')
    model_settings.add_argument('--num_private_train_splits', type=int, default=20,
                                help='num_private_train_splits=M')
    model_settings.add_argument('--K', type=int, default=100,
                                help='Vocabulary size')
    model_settings.add_argument('--t', type=float, default=0.8,
                                help='Target number of points in the ball = t * M')
    model_settings.add_argument('--alpha_r', type=float, default=1.65,
                                help='alpha for dp-radius (alpha,tau)-RDP')
    model_settings.add_argument('--tau_r', type=float, default=0.005,
                                help='tau for dp-radius (alpha,tau)-RDP')
    model_settings.add_argument('--sigma_test', type=float, default=2.819,
                                help='Noise scale for DP-testing')
    model_settings.add_argument('--llambda_test', type=float, default=0.95,
                                help='lambda for DP-testing')
    model_settings.add_argument('--llambda_avg', type=float, default=0.5,
                                help='lambda for DP-avg')
    model_settings.add_argument('--sigma_avg', type=float, default= 0.770,
                                help='Noise scale for DP-avg')
    model_settings.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf",
                                help='Language model')
    model_settings.add_argument('--T_max', type=int, default=20,
                                help='Number of tokens')
    model_settings.add_argument('--T_update', type=int, default=2,
                                help='Number of iterations of updating radius')
    model_settings.add_argument('--noise_seed', type=int, default=0,
                                help='Noise seed')
    model_settings.add_argument('--sample_seed', type=int, default=0,
                                help='Sample seed')
    return parser.parse_args()

def construct_prompt_same(train_examples, test_example):
    # Prompt for testing
    prompt = ""
    for train_example in train_examples:
        prompt += "Sentence: " + train_example["text"] + "\n"
        prompt += "Genre: " + train_example["label"] + "\n\n"
    prompt += "Sentence: " + test_example["text"] + "\n"
    prompt += "Genre:"
    return prompt

def construct_prompt_different(example, random_label, existing_answer):
    # Prompt for generating synthetic data
    prompt = "Given a genre for the film, generate a description accordingly and make sure to include the given genre in the description.\n\n"
    for example in example:
        prompt += "Genre: " + example["label"] + "\n"
        prompt += "Sentence: " + example["text"] + "\n\n"
    prompt += "Genre: " + random_label + "\n"
    prompt += "Sentence: " + existing_answer
    return prompt

def complete(prompt, l, model_name, temp=0, num_log_probs=None, echo=False, n=None):
    # Query LLM for soft label
    response = client.completions.create(
        model="meta-llama/Llama-2-7b-hf",
        prompt=prompt,
        max_tokens=l,
        #temperature=temp,
        logprobs=num_log_probs,
        echo=echo,
        stop="\n",
        #n=n,
    )
    return response.choices[0]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def query_vllm(prompt):
    # Query LLM for testing
    completion = client.completions.create(
        model="meta-llama/Llama-2-7b-hf",
        prompt=prompt,
        max_tokens=10,
        #logprobs=10000,
    )
    #print(completion.choices[0])
    predicted_label = completion.choices[0].text.strip().split("\n")[0].strip()

    return predicted_label



def test_all_examples(syn_examples, test_examples):
    # Testing
    predictions = []

    for test_example in test_examples:
        prompt = construct_prompt_same(syn_examples, test_example)
        #print(prompt)
        predicted_label_text = query_vllm(prompt)
        #print(predicted_label_text)
        predictions.append(predicted_label_text)

    true_labels = [example['label'] for example in test_examples]

    # Calculate accuracy
    correct_predictions = sum(1 for true_label, pred_label in zip(true_labels, predictions) if true_label == pred_label)
    accuracy = 100 * correct_predictions / len(test_examples)
    return accuracy

def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def count_points_within_radius(S, p, r):
    """Count how many points from S are within radius r from point p."""
    count = 0
    for point in S:
        if distance(point, p) <= r:
            count += 1
    return count

def B_r(S, p, r, t):
    """Calculate the capped number of points within radius r from point p."""
    return min(count_points_within_radius(S, p, r), t)

def L(S, r, t):
    """Compute L(r) for the dataset S."""
    counts = [B_r(S, p, r, t) for p in S]
    largest_t_counts = sorted(counts, reverse=True)[:t]
    return sum(largest_t_counts) / t

def clip_points_to_ball(points, center, radius):
    """
    Clip each point in points to ensure it lies within a ball of radius centered at c.
    """
    clipped_points = []
    for p in points:
        displacement = p - center
        distance = np.linalg.norm(displacement)
        if distance > radius:
            # Scale the point to lie on the boundary of the ball
            scaled_displacement = displacement * radius / distance
            clipped_point = center + scaled_displacement
        else:
            # Leave the point unchanged if it's inside the ball
            clipped_point = p
        clipped_points.append(clipped_point)
    return np.array(clipped_points)

def dp_good_radius(S, t, tolerance, alpha, tau, seed):
    # Find the radius r, such that (1) (alpha, tau)-RDP (2) there exists a ball of radius r covering t percent points
    # Following the GoodRadius algorithm in https://arxiv.org/pdf/1604.05590.pdf
    r_low, r_high = 0, np.sqrt(2)/2
    np.random.seed(seed)

    max_iterations = math.ceil(math.log2((r_high - r_low) / tolerance))
    delta = 1e-4
    epsilon_gaussian = RDP_DP(alpha, tau / (2 * max_iterations), delta)
    sigma = np.sqrt(2 * np.log(1.25/delta))/epsilon_gaussian

    while r_high - r_low > tolerance:
        r_mid = (r_high + r_low) / 2
        l_mid = L(S, r_mid, t) + np.random.normal(0, 2 * sigma)
        l_half_mid = L(S, r_mid / 2, t) + np.random.normal(0, 2 * sigma)

        if l_half_mid >= t:
            r_high = r_mid
        else:
            if l_mid >= t: # Satisfied
                r_high = r_mid
            else:
                r_low = r_mid

    return (r_high + r_low) / 2

def RDP_DP(alpha, gamma, delta):
    # If an algorithm satisfies (optimal_epsilon, delta)-DP, then it satisfies (alpha, gamma)-RDP.
    # Function to find epsilon given alpha, gamma, and delta
    # Following Theorem 3 in https://arxiv.org/pdf/2008.06529.pdf
    if alpha<=1:
        print("Error: alpha<=1")
    def objective(epsilon):
        def gamma_alpha(p):
            M_value = (p ** alpha) * ((p - delta) ** (1 - alpha)) + ((1 - p) ** alpha) * ((math.exp(epsilon) - p + delta) ** (1 - alpha))
            gamma = epsilon + (1 / (alpha - 1)) * np.log(M_value)
            return gamma

        res = minimize_scalar(lambda p: gamma_alpha(p), bounds=(delta, 1), method='bounded')
        if not res.success:
            raise ValueError("Optimization did not converge inside gamma calculation.")

        gamma_calc = res.fun
        return gamma_calc - gamma

    # Use root_scalar to find the epsilon that makes objective(epsilon) = 0
    result = root_scalar(objective, bracket=[0, 100], method='brentq')  # Assuming epsilon is in the range [0, 100]

    if not result.converged:
        raise ValueError("Root finding did not converge.")

    optimal_epsilon = result.root

    return optimal_epsilon

if __name__ == "__main__":

    args = parse_args()
    # num_private_train=MN
    num_private_train = args.num_private_train
    # num_valid=n. n samples to be generated for n-shot ICL
    num_valid = args.n_shot
    # num_private_train_splits=M
    num_private_train_splits = args.num_private_train_splits
    # Vocabulary size=K
    K = args.K
    model_name = args.model_name
    T_max = args.T_max
    noise_seed = args.noise_seed
    sample_seed = args.sample_seed
    # Target number of points in the ball = t * M
    t = args.t
    # RDP requirements for DP-radius
    alpha_r = args.alpha_r
    tau_r = args.tau_r
    # Noise scale for DP-testing
    sigma_test = args.sigma_test
    # lambda for DP-testing
    llambda_test = args.llambda_test
    # lambda for DP-avg
    llambda_avg = args.llambda_avg
    # Noise scale for DP-avg
    sigma_avg = args.sigma_avg
    T_update = args.T_update

    DEFAULT_NUM_TEST = 780

    train_sentences, train_labels, test_sentences, test_labels = load_movie_genre()
    train_data = [{'text': text, 'label': label} for text, label in zip(train_sentences, train_labels)]

    random.seed(sample_seed)
    selected_labels = random.sample(train_labels, num_valid)


    # Generate 4 examples (4-shot)
    syn_data = []

    for n_shot in range(num_valid):
        existing_answer = ""
        for i in range(T_max):
            # print(i)
            random.seed(sample_seed)
            # Sample M*N data
            selected_private_data = random.sample(train_data, num_private_train)
            splits = list(chunks(selected_private_data, int(num_private_train / num_private_train_splits)))
            all_logprobs = []
            # Turn the data into M prompts
            for split in splits:
                prompt = construct_prompt_different(split, selected_labels[n_shot], existing_answer)
                #print(prompt)
                raw_reponse = complete(prompt, 1, model_name, temp=0, num_log_probs=20000, echo=False, n=None)
                top_logprobs = raw_reponse.logprobs.top_logprobs[0]
                normalized_top_logprobs = {key.replace('0x0A>', ''): value for key, value in top_logprobs.items()}
                all_logprobs.append(normalized_top_logprobs)

            private_prompt = construct_prompt_different([], selected_labels[n_shot], existing_answer)
            private_raw_reponse = complete(private_prompt, 1, model_name, temp=0, num_log_probs=K, echo=False, n=None)
            private_top_logprobs = private_raw_reponse.logprobs.top_logprobs[0]
            private_normalized_top_logprobs = {key.replace('0x0A>', ''): value for key, value in private_top_logprobs.items()}

            filtered_logprobs = []
            for logprob_dict in all_logprobs:
                filtered_dict = {key: logprob_dict[key] for key in private_normalized_top_logprobs if key in logprob_dict}
                filtered_logprobs.append(filtered_dict)
            all_logprobs = filtered_logprobs
            keys = list(filtered_logprobs[0].keys())
            points = np.array([[np.exp(logprob_dict[key]) if key in logprob_dict else 0 for key in keys] for logprob_dict in filtered_logprobs])

            # Get desired DP-radius r
            r = dp_good_radius(points, int(t * num_private_train_splits), 0.1, alpha_r, tau_r, noise_seed)
            #print(f"Good Radius: {round(r,3)}")

            # Current radius R
            R = np.sqrt(2)/2
            # Clipping center
            clipped_points = points
            np.random.seed(noise_seed)
            for j in range(T_update):
                # DP-avg
                sum_clipped_point = np.sum(clipped_points, axis=0)
                noise_level = 2 * R * sigma_avg
                noise = np.random.normal(0, noise_level, size=sum_clipped_point.shape)
                avg_clipped_point_noise = (noise+sum_clipped_point)/num_private_train_splits
                # check if R can decrease
                if R < r + llambda_avg * np.sqrt(K) * noise_level / num_private_train_splits:
                    break
                #print("R can decrease")
                # DP-testing
                covered_points = count_points_within_radius(points, avg_clipped_point_noise, r + llambda_test * np.sqrt(K) * noise_level/num_private_train_splits)
                #print(f"covered points: {covered_points}")
                if covered_points + np.random.normal(0, sigma_test, size=1) < 0.7 * num_private_train_splits:
                    break
                # Update R
                R = r + llambda_avg * np.sqrt(K) * noise_level / num_private_train_splits
                #print(f"Update R in iter {i}: {round(R,3)}")
                # Clip points into a ball with radius R and center
                clipped_points = clip_points_to_ball(points,avg_clipped_point_noise,R)

            max_dim_index = np.argmax(avg_clipped_point_noise)
            max_key = keys[max_dim_index]
            existing_answer += max_key

        existing_answer = existing_answer.replace("▁", " ")
        #print(selected_labels[n_shot])
        #print(existing_answer)
        syn_data.append({'text': existing_answer, 'label': selected_labels[n_shot]})

    # Test
    test_examples = [{'text': text, 'label': label} for text, label in zip(test_sentences, test_labels)]
    random.seed(12345)
    random.shuffle(test_examples)
    test_subset = test_examples[:DEFAULT_NUM_TEST]

    accuracy = test_all_examples(syn_data, test_subset)
    print(f"Accuracy: {accuracy:.3f}")


