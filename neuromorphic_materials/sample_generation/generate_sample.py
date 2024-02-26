#! /usr/bin/env python3
from src import generate
from src.arg_parser import GenerateSampleParser


def main() -> None:
    # TODO: Refactor to use sub-parsers when a version
    #  with typed subparser support is released
    base_args = GenerateSampleParser(add_help=False).parse_args(known_only=True)

    generator_type = base_args.sample_type.generator_type()
    args = base_args.sample_type.arg_parser().parse_args(base_args.extra_args)

    if args.num_samples > 1:
        generate.bulk(generator_type, args)
    else:
        generate.single(generator_type, args)


if __name__ == "__main__":
    main()
