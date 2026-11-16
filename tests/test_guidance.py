"""Distribution and sampler integration checks; no trained weights required."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from torch import nn

# Direct execution puts tests/, rather than the repository root, on sys.path.
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generate
from protgen.data import total_tokens
from protgen.guidance import (
    guide_logits,
    load_prefix_classifier,
    validate_guidance,
)
from protgen.models import GRU, MLP, FamilyClassifier


CPU = torch.device("cpu")
PAD, BOS, EOS = (total_tokens.index(token) for token in ("<PAD>", "<BOS>", "<EOS>"))
A, C = (total_tokens.index(token) for token in ("A", "C"))


class CandidateClassifier(nn.Module):
    """p(HU | prefix+A)=1/4, p(HU | prefix+C)=3/4; shuffled labels."""

    fam_to_idx = {"HPr": 0, "HU": 1, "AcyP": 2}

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, tokens):
        self.seen.append(tokens.clone())
        score = torch.where(tokens[:, -1] == A, 0.25, 0.75)
        return torch.stack(((1 - score) / 2, score, (1 - score) / 2), dim=-1).log()


class ForbiddenClassifier(nn.Module):
    def eval(self):
        raise AssertionError("lambda=0 must not change classifier mode")

    @property
    def fam_to_idx(self):
        raise AssertionError("lambda=0 must not read classifier labels")

    def forward(self, tokens):
        raise AssertionError("lambda=0 must not run classifier inference")


class GuidanceTests(unittest.TestCase):
    def test_zero_strength_matches_original_probabilities_for_same_prefix(self):
        """Observe distributions passed to multinomial, with fixed prefix tokens.

        The reference repeats the pre-guidance model/temperature/mask operations.
        Forced draws ensure every comparison uses exactly the same model/prefix.
        """
        for model_class in (GRU, MLP):
            for fixed_length in (False, True):
                for strength in (None, 0.0):
                    with self.subTest(model=model_class.__name__, fixed=fixed_length,
                                      strength=strength):
                        torch.manual_seed(7)
                        model = model_class(vocab_size=len(total_tokens), hidden_dim=8).eval()
                        expected = []
                        hidden = None
                        context = torch.tensor([[PAD, PAD, BOS]])
                        token = torch.tensor([BOS])
                        with torch.inference_mode():
                            for forced in (A, C, A):
                                if isinstance(model, GRU):
                                    logits, hidden = model.step(token, hidden)
                                else:
                                    logits = model(context)[:, -1, :]
                                logits = logits / 0.7
                                logits[:, [PAD, BOS]] = -torch.inf
                                if fixed_length:
                                    logits[:, EOS] = -torch.inf
                                expected.append(torch.softmax(logits, dim=-1).clone())
                                token = torch.tensor([forced])
                                context = torch.cat((context[:, 1:], token[:, None]), dim=1)

                        observed = []

                        def draw(probabilities, num_samples):
                            self.assertEqual(num_samples, 1)
                            observed.append(probabilities.clone())
                            return torch.tensor([[(A, C, A)[len(observed) - 1]]])

                        kwargs = {} if strength is None else {"guidance_strength": strength}
                        with patch("generate.torch.multinomial", side_effect=draw), patch(
                            "generate.guide_logits", side_effect=AssertionError("must bypass guidance")
                        ):
                            sequences = generate.sample_sequences(
                                model, 1, 3, 0.7, CPU,
                                target_lengths=[3] if fixed_length else None,
                                classifier=ForbiddenClassifier(), **kwargs,
                            )
                        self.assertEqual(sequences, ["ACA"])
                        self.assertEqual(len(observed), len(expected))
                        for before, after in zip(expected, observed):
                            self.assertTrue(torch.equal(before, after))

    def test_zero_strength_helper_does_not_access_classifier(self):
        logits = torch.randn(1, len(total_tokens))
        self.assertIs(guide_logits(logits, None, ForbiddenClassifier(), None, 0.0, None), logits)

    def test_candidate_guidance_has_expected_normalized_distribution(self):
        logits = torch.full((1, len(total_tokens)), -torch.inf)
        logits[0, [A, C]] = torch.tensor([0.6, 0.4]).log()
        prefixes = torch.tensor([[BOS, C]])
        for strength, expected in ((1.0, [1 / 3, 2 / 3]), (2.0, [1 / 7, 6 / 7])):
            with self.subTest(strength=strength):
                classifier = CandidateClassifier()
                guided = guide_logits(logits, prefixes, classifier, "HU", strength,
                                      torch.tensor([True]))
                probabilities = guided.softmax(-1)
                torch.testing.assert_close(probabilities[0, [A, C]], torch.tensor(expected))
                torch.testing.assert_close(probabilities.sum(-1), torch.ones(1))
                self.assertTrue(torch.equal(torch.isneginf(logits), torch.isneginf(guided)))
                self.assertTrue(torch.equal(torch.cat(classifier.seen),
                                            torch.tensor([[BOS, C, A], [BOS, C, C]])))

    def test_inactive_rows_are_unchanged_and_inference_is_chunked(self):
        # 14 * 20 = 280 candidates exceeds one 256-candidate inference chunk.
        logits = torch.randn(15, len(total_tokens))
        logits[:, [PAD, BOS, EOS]] = -torch.inf
        prefixes = torch.tensor([[BOS, A]]).expand(15, -1)
        active = torch.tensor([True] * 14 + [False])
        classifier = CandidateClassifier()
        guided = guide_logits(logits, prefixes, classifier, "HU", 1.0, active)
        self.assertEqual([batch.size(0) for batch in classifier.seen], [256, 24])
        self.assertTrue(torch.equal(guided[-1], logits[-1]))
        inputs = torch.cat(classifier.seen)
        self.assertTrue(torch.equal(inputs[:, :2], prefixes[:1].expand(280, -1)))
        self.assertTrue((inputs[:, -1] >= A).all())

    def test_sampler_normalizes_guidance_after_temperature_and_masks(self):
        model = GRU(vocab_size=len(total_tokens), hidden_dim=8)
        logits = torch.full((1, len(total_tokens)), -torch.inf)
        # At temperature 2, generator weights become (0.6, 0.4).
        logits[0, [A, C]] = 2 * torch.tensor([0.6, 0.4]).log()
        logits[0, [PAD, BOS, EOS]] = 100.0
        classifier = CandidateClassifier()
        with patch.object(model, "step", return_value=(logits, None)), patch(
            "generate.torch.multinomial", return_value=torch.tensor([[A]])
        ) as draw:
            generate.sample_sequences(
                model, 1, 1, 2.0, CPU, target_lengths=[1],
                target_family="HU", classifier=classifier, guidance_strength=1.0,
            )
        probabilities = draw.call_args.args[0]
        torch.testing.assert_close(probabilities[0, [A, C]], torch.tensor([1 / 3, 2 / 3]))
        self.assertEqual(probabilities[0, [PAD, BOS, EOS]].tolist(), [0.0, 0.0, 0.0])
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(1))
        self.assertTrue(torch.equal(torch.cat(classifier.seen), torch.tensor([[BOS, A], [BOS, C]])))

    def test_guided_sampler_uses_full_prefix_and_preserves_length_stopping(self):
        model = GRU(vocab_size=len(total_tokens), hidden_dim=8)
        classifier = CandidateClassifier()
        draws = [torch.tensor([[A], [C]]), torch.tensor([[C], [A]])]
        with patch("generate.torch.multinomial", side_effect=draws):
            sequences = generate.sample_sequences(
                model, 2, 9, 1.0, CPU, target_lengths=[1, 2],
                guidance_strength=1.0, classifier=classifier, target_family="HU",
            )
        self.assertEqual(sequences, ["A", "CA"])
        self.assertEqual([batch.size(0) for batch in classifier.seen], [40, 20])
        self.assertTrue(torch.equal(classifier.seen[1][:, :2],
                                    torch.tensor([[BOS, C]]).expand(20, -1)))
        for batch in classifier.seen:
            self.assertTrue((batch[:, -1] >= A).all())

    def test_guided_free_length_scores_eos_and_stops(self):
        model = GRU(vocab_size=len(total_tokens), hidden_dim=8)
        classifier = CandidateClassifier()
        with patch("generate.torch.multinomial", return_value=torch.tensor([[EOS]])) as draw:
            sequences = generate.sample_sequences(
                model, 1, 5, 1.0, CPU, guidance_strength=1.0,
                classifier=classifier, target_family="HU",
            )
        self.assertEqual(sequences, [""])
        draw.assert_called_once()
        self.assertTrue((classifier.seen[0][:, -1] == EOS).any())
        self.assertTrue((classifier.seen[0][:, -1] >= EOS).all())

    def test_checkpoint_adapter_matches_person_1_inference(self):
        config = dict(vocab_size=len(total_tokens), hidden_dim=8, dropout=0.2)
        head_config = dict(hidden_dim=8, num_classes=3, head_hidden_dim=5, dropout=0.2)
        backbone, head = GRU(**config).eval(), FamilyClassifier(**head_config).eval()
        mapping = {"HPr": 0, "HU": 1, "AcyP": 2}
        tokens = torch.tensor([[BOS, A, C], [BOS, C, A]])
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "classifier.pt"
            torch.save(dict(backbone_config=config, backbone_state_dict=backbone.state_dict(),
                            head_config=head_config, head_state_dict=head.state_dict(),
                            fam_to_idx=mapping), checkpoint)
            classifier = load_prefix_classifier(checkpoint, CPU)
        with torch.inference_mode():
            expected = head(backbone.hidden_states(tokens)[:, -1, :])
            self.assertTrue(torch.equal(classifier(tokens), expected))
        self.assertEqual(classifier.fam_to_idx, mapping)
        self.assertFalse(classifier.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in classifier.parameters()))

    def test_invalid_guidance_is_rejected(self):
        classifier = CandidateClassifier()
        for strength, family, candidate in (
            (-1.0, "HU", classifier), (float("nan"), "HU", classifier),
            (float("inf"), "HU", classifier), (1.0, None, classifier),
            (1.0, "unknown", classifier), (1.0, "HU", None),
        ):
            with self.subTest(strength=strength, family=family), self.assertRaises(ValueError):
                validate_guidance(strength, family, candidate)
        classifier.fam_to_idx = {"AcyP": 0, "HPr": 0, "HU": 2}
        with self.assertRaises(ValueError):
            validate_guidance(1.0, "HU", classifier)

    def test_cli_requires_classifier_only_for_positive_strength(self):
        for argv in ([], ["--lambda", "0"], ["--guidance-strength", "0"]):
            with self.subTest(argv=argv), patch.object(sys, "argv", ["generate.py", *argv]):
                self.assertEqual(generate.parse_args().guidance_strength, 0.0)
        for argv in (["--lambda", "1"], ["--lambda", "-1"], ["--lambda", "nan"]):
            with self.subTest(argv=argv), patch.object(sys, "argv", ["generate.py", *argv]), \
                    patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                generate.parse_args()
            self.assertEqual(error.exception.code, 2)
        with patch.object(sys, "argv", ["generate.py", "--lambda", "1", "--target-family", "HU",
                                        "--classifier-checkpoint", "classifier.pt"]):
            self.assertEqual(generate.parse_args().guidance_strength, 1.0)

    def test_zero_strength_main_never_loads_a_classifier(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model_config = dict(vocab_size=len(total_tokens), pad_token_id=PAD, hidden_dim=8)
            model = GRU(**model_config)
            torch.save(dict(model_type="gru", model_config=model_config,
                            model_state_dict=model.state_dict()), root / "gru.pt")
            argv = ["generate.py", "--checkpoint", str(root / "gru.pt"), "--device", "cpu",
                    "--num-samples", "1", "--free-length", "--max-length", "2",
                    "--classifier-checkpoint", str(root / "does-not-exist.pt"),
                    "--out", str(root / "samples.csv")]
            with patch.object(sys, "argv", argv), patch("builtins.print"), patch(
                "generate.load_prefix_classifier", side_effect=AssertionError("must not load")
            ):
                generate.main()
            self.assertTrue((root / "samples.csv").is_file())


if __name__ == "__main__":
    unittest.main()
