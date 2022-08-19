from ensemble_projection.bayes_inference import bayes_infer
from args import Args
import cProfile


def main():
    args = Args().parse_args()
    bayes_infer(**args.as_dict())

# cProfile.run("main()", sort="cumulative")

if __name__ == "__main__":
    main()