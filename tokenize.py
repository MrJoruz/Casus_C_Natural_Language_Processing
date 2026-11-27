#!/usr/bin/env python3
"""Byte-pair encoding tokenizer (les 1, casus NLP).

Voorbeelden:
    tokenize train kanker_nl.txt --vocab-size 1000
    tokenize train kanker_nl.txt -o nl.enc --encode
    tokenize encode kanker_en.txt -e nl.enc -o en_met_nl.tok
    tokenize decode en_met_nl.tok -e nl.enc -o terug.txt
"""

import argparse
import sys
from pathlib import Path

import nlp


def parse_args(argv=None):
    """Bouw de command-line interface, lees de argumenten in en controleer ze."""
    parser = argparse.ArgumentParser(
        prog="tokenize",
        description="Leer een byte-pair encoding uit tekst (.txt -> .enc), "
                    "zet tekst om in tokens (.txt -> .tok) of tokens terug in "
                    "tekst (.tok -> .txt).",
        epilog="Gebruik 'tokenize <commando> -h' voor de opties per commando.",
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="toon voortgang en statistieken op stderr")
    sub = parser.add_subparsers(dest="command", metavar="<commando>", required=True)

    # --- train ------------------------------------------------------------
    train = sub.add_parser(
        "train",
        help="leer een byte-pair encoding uit een of meer tekstbestanden",
        description="Leer een byte-pair encoding uit een of meer .txt-bestanden "
                    "en schrijf die naar een .enc-bestand. Het samenvoegen stopt "
                    "zodra --vocab-size bereikt is of geen paar meer "
                    "--min-freq keer voorkomt.",
    )
    train.add_argument("inputs", nargs="+", type=Path, metavar="TXT",
                       help="tekstbestand(en) om de encoding uit te leren")
    train.add_argument("-o", "--output", type=Path, metavar="ENC",
                       help="uitvoerbestand (standaard: <eerste invoer>.enc)")
    train.add_argument("-n", "--vocab-size", type=int, default=1000, metavar="N",
                       help="maximaal aantal verschillende tokens (standaard: %(default)s)")
    train.add_argument("-m", "--min-freq", type=int, default=2, metavar="K",
                       help="minimale frequentie van een paar om samen te voegen "
                            "(standaard: %(default)s)")
    train.add_argument("-l", "--lowercase", action="store_true",
                       help="zet alle tekst om naar kleine letters")
    train.add_argument("-p", "--punctuation", choices=["keep", "strip"], default="keep",
                       help="interpunctie als eigen tokens behouden of "
                            "verwijderen (standaard: %(default)s); interpunctie "
                            "wordt nooit samengevoegd met letters")
    train.add_argument("--encode", action="store_true",
                       help="zet de invoerbestanden direct ook om naar .tok "
                            "(naast elk invoerbestand)")

    # encode
    encode = sub.add_parser(
        "encode",
        help="zet tekst om in tokens met een bestaande encoding",
        description="Pas een .enc-encoding toe op een of meer .txt-bestanden en "
                    "schrijf per bestand een .tok-bestand.",
    )
    encode.add_argument("inputs", nargs="+", type=Path, metavar="TXT",
                        help="tekstbestand(en) om om te zetten")
    encode.add_argument("-e", "--encoding", type=Path, required=True, metavar="ENC",
                        help="byte-pair encoding (.enc)")
    encode.add_argument("-o", "--output", type=Path, metavar="TOK",
                        help="uitvoerbestand; alleen bij precies een invoer "
                             "(standaard: <invoer>.tok)")

    # decode
    decode = sub.add_parser(
        "decode",
        help="zet tokens terug om in tekst",
        description="Pas een .enc-encoding toe op een of meer .tok-bestanden en "
                    "schrijf per bestand de tekst terug naar een .txt-bestand.",
    )
    decode.add_argument("inputs", nargs="+", type=Path, metavar="TOK",
                        help="tokenbestand(en) om terug te zetten")
    decode.add_argument("-e", "--encoding", type=Path, required=True, metavar="ENC",
                        help="byte-pair encoding (.enc)")
    decode.add_argument("-o", "--output", type=Path, metavar="TXT",
                        help="uitvoerbestand; alleen bij precies een invoer "
                             "(standaard: <invoer>.txt, of <invoer>.decoded.txt "
                             "als dat de invoer zou overschrijven)")

    args = parser.parse_args(argv)

    # controles die argparse zelf niet doet
    files = list(args.inputs) + ([args.encoding] if args.command != "train" else [])
    for path in files:
        if not path.is_file():
            parser.error(f"bestand '{path}' bestaat niet")
    if args.command == "train":
        if args.vocab_size < 2:
            parser.error("--vocab-size moet minstens 2 zijn")
        if args.min_freq < 1:
            parser.error("--min-freq moet minstens 1 zijn")
    elif args.output and len(args.inputs) > 1:
        parser.error("-o/--output kan alleen gebruikt worden met precies een invoerbestand")

    return args


def main(argv=None):
    args = parse_args(argv)

    def log(message):
        if args.verbose:
            print(message, file=sys.stderr)

    if args.command == "train":
        text = "\n".join(path.read_text(encoding="utf-8") for path in args.inputs)
        tokenizer = nlp.BPETokenizer(lowercase=args.lowercase,
                                     punctuation=args.punctuation)
        tokenizer.fit(text, max_tokens=args.vocab_size, min_freq=args.min_freq)
        output = args.output or args.inputs[0].with_suffix(".enc")
        tokenizer.save(output)
        log(f"{len(tokenizer)} tokens geleerd -> {output}")
        if args.encode:
            for path in args.inputs:
                tokens = tokenizer.encode(path.read_text(encoding="utf-8"))
                nlp.write_tokens(path.with_suffix(".tok"), tokens)
                log(f"{path}: {len(tokens)} tokens -> {path.with_suffix('.tok')}")

    elif args.command == "encode":
        tokenizer = nlp.BPETokenizer.load(args.encoding)
        for path in args.inputs:
            tokens = tokenizer.encode(path.read_text(encoding="utf-8"))
            output = args.output or path.with_suffix(".tok")
            nlp.write_tokens(output, tokens)
            log(f"{path}: {len(tokens)} tokens -> {output}")

    elif args.command == "decode":
        tokenizer = nlp.BPETokenizer.load(args.encoding)
        for path in args.inputs:
            text = tokenizer.decode(nlp.read_tokens(path))
            output = args.output or path.with_suffix(".txt")
            if output.resolve() == path.resolve():
                output = path.with_suffix(".decoded.txt")
            output.write_text(text, encoding="utf-8")
            log(f"{path}: {len(text)} tekens -> {output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
