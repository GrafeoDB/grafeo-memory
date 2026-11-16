"""Quick benchmark for temporal knowledge graph features.

Exercises:
  1. Bi-temporal model (valid_at/invalid_at, point-in-time queries)
  2. Episode provenance (Episode nodes, PRODUCED edges, session replay)

Requires:
    MISTRAL_API_KEY in .env
"""

from __future__ import annotations

import time

from dotenv import load_dotenv

load_dotenv()

from mistralai.client import Mistral  # noqa: E402

from grafeo_memory import MemoryConfig, MemoryManager, MistralEmbedder  # noqa: E402


def main():
    client = Mistral()
    embedder = MistralEmbedder(client)

    config = MemoryConfig(
        user_id="monica",
        enable_bitemporal=True,
        enable_episodes=True,
        run_id="bench_run_1",
    )

    t0 = time.perf_counter()

    with MemoryManager("mistral:mistral-small-latest", config, embedder=embedder) as mem:
        # ------------------------------------------------------------------
        # Phase 1: Bi-temporal -- add facts with real-world dates
        # ------------------------------------------------------------------
        print("=== Phase 1: Bi-Temporal ===\n")

        msgs = [
            "Monica joined Raviga Capital as an associate in March 2014.",
            "Monica started working with Pied Piper for Raviga in April 2015.",
            "Monica left Raviga Capital in March 2018 and founded Bream-Hall with Laurie Bream as a partner.",
            "Monica has been friends with Richard Hendricks since 2014.",
        ]

        for msg in msgs:
            t1 = time.perf_counter()
            events = mem.add(msg)
            dt = (time.perf_counter() - t1) * 1000
            for e in events:
                va = f"  valid_at={e.valid_at}" if e.valid_at else ""
                print(f"  [{e.action.value.upper():6s}] {e.text}{va}")
                if e.old_text:
                    print(f"           was: {e.old_text}")
            print(f"  ({dt:.0f}ms)\n")

        # Point-in-time search: "as of mid-2016"
        from datetime import UTC, datetime

        mid_2016 = int(datetime(2016, 7, 1, tzinfo=UTC).timestamp() * 1000)
        print("Search: 'Where does Monica work?' (point_in_time=2016-07-01)")
        results = mem.search("Where does Monica work?", point_in_time=mid_2016)
        for r in results[:3]:
            va = f"  valid_at={r.valid_at}" if r.valid_at else ""
            ia = f"  invalid_at={r.invalid_at}" if r.invalid_at else ""
            print(f"  [{r.score:.2f}] {r.text}{va}{ia}")
        print()

        # Current search (no point_in_time)
        print("Search: 'Where does Monica work?' (current)")
        results = mem.search("Where does Monica work?")
        for r in results[:3]:
            va = f"  valid_at={r.valid_at}" if r.valid_at else ""
            print(f"  [{r.score:.2f}] {r.text}{va}")
        print()

        # ------------------------------------------------------------------
        # Phase 2: Episode provenance
        # ------------------------------------------------------------------
        print("=== Phase 2: Episode Provenance ===\n")

        episodes = mem.get_episodes()
        print(f"Episodes created: {len(episodes)}")
        for ep in episodes:
            print(f"  [{ep.episode_id}] {ep.content[:60]}...")
            print(f"    produced: {ep.produced_memories}")
            print(f"    mentions: {ep.mentioned_entities}")
        print()

        # Session replay
        if episodes:
            print("Episode chain (forward from first):")
            chain = mem.episode_chain(episodes[0].episode_id, direction="forward")
            for ep in chain:
                print(f"  -> [{ep.episode_id}] {ep.content[:50]}...")
            print()

            # Provenance
            memories = mem.get_all()
            if memories:
                mid = memories[0].memory_id
                prov = mem.get_provenance(mid)
                print(f"Provenance for memory '{memories[0].text[:40]}...':")
                for ep in prov:
                    print(f"  <- Episode [{ep.episode_id}] {ep.content[:50]}...")
                print()

        # ------------------------------------------------------------------
        # Stats
        # ------------------------------------------------------------------
        print("=== Stats ===\n")
        stats = mem.stats()
        print(f"  Memories:    {stats.total_memories}")
        print(f"  Entities:    {stats.entity_count}")
        print(f"  Relations:   {stats.relation_count}")
        print(f"  Episodes:    {stats.episode_count}")
        print(f"  Communities: {stats.community_count}")
        print()

    total = (time.perf_counter() - t0) * 1000
    print(f"Total time: {total:.0f}ms")


if __name__ == "__main__":
    main()
