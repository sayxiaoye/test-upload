from src.rag_cli import build_parser


def test_build_parser_supports_separate_retrieve_and_rerank_limits():
    parser = build_parser()
    args = parser.parse_args(
        [
            "难看",
            "--doc-file",
            "data/jp",
            "--retrieve-k",
            "7",
            "--rerank-k",
            "2",
        ]
    )

    assert args.retrieve_k == 7
    assert args.rerank_k == 2
