using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Models.Relics;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using SpireHost;
using SpireHost.Driver;
using SpireHost.Driver.Tools;

namespace SpireHost.Driver.Events;

/// <summary>
/// --selftest [--events A,B|all] [--variants base,poor,...] [--seeds N] [--hp N] [--skip-cases] [--skip-sweep] [--brief] [--out FILE]
///
/// Part 1 (cases): targeted scenarios that prove each frame / patch and that the game logic after every patched
///   presentation call still runs (bundle pick + skip with the choice-id protocol, Crystal Sphere, Trial confirm
///   NO / YES, FakeMerchant buy + FoulPotion combat, potion-lock, WaxChoker ancient proceed, TheArchitect win, lethal
///   option, the seven fuzz-found event sites).
/// Part 2 (sweep): every event type of the game (71 = 73 minus the two Deprecated* placeholders) x state variants
///   (base / poor / hurt / lethal / rich; act raised until EventModel.IsAllowed holds) x seeds x characters, played
///   with a coverage-directed policy (least-chosen option first) until the event is left for the map. Gates: no
///   abort, every option that was ever legal was chosen at least once, per-decision invariants.
/// </summary>
public static class EventsCommands
{
    public static int Run(HostSession session, Args a)
    {
        if (a.Command == "list-event-patches")
        {
            HostJson.Emit(new
            {
                kind = "events-patches",
                patches = SpireHost.Patches.PatchRegistry.Records.Where(r => r.Id.StartsWith("events-", StringComparison.Ordinal))
                    .Select(r => new { id = r.Id, target = r.Target, kind = r.Kind, category = r.Category.ToString(), reason = r.Reason }).ToList(),
                all_by_category = SpireHost.Patches.PatchRegistry.Records.GroupBy(r => r.Category.ToString()).ToDictionary(g => g.Key, g => g.Count()),
            }, a.Get("out"));
            return 0;
        }
        if (a.Command != "selftest")
        {
            Console.Error.WriteLine("events: --selftest [--events A,B|all] [--variants ..] [--seeds N] [--extra-rounds N] [--hp N] [--skip-cases] [--skip-sweep] [--brief] | --list-event-patches");
            return 1;
        }
        return SelfTest.Run(session, a);
    }
}

internal static class SelfTest
{
    public static int Run(HostSession session, Args a)
    {
        Stopwatch wall = Stopwatch.StartNew();
        int hp = a.GetInt("hp", 9999);
        List<object> caseResults = new List<object>();
        int casesPassed = 0, casesTotal = 0;
        if (!a.Has("skip-cases"))
        {
            foreach (Case c in Cases.All())
            {
                for (int s = 0; s < c.Seeds; s++)
                {
                    casesTotal++;
                    (bool caseOk, object summary) = Cases.Run(session, c, s, hp);
                    if (caseOk)
                    {
                        casesPassed++;
                    }
                    caseResults.Add(summary);
                }
            }
        }

        Sweep.Result? sweep = null;
        if (!a.Has("skip-sweep"))
        {
            sweep = Sweep.Run(session, a, hp);
        }

        bool ok = casesPassed == casesTotal && (sweep == null || sweep.Ok);
        HostJson.Emit(new
        {
            kind = "events-selftest",
            ok,
            event_types_in_game = EventIds.CountTypes(),
            event_handler = session.Registry.GetRoomHandler(RoomScreen.Event)?.Name,
            cases_passed = casesPassed,
            cases_total = casesTotal,
            sweep_ok = sweep?.Ok,
            sweep = sweep?.Summary(a.Has("brief")),
            patch_hits = new SortedDictionary<string, long>(EventPatches.Counts.ToDictionary(kv => kv.Key, kv => kv.Value), StringComparer.Ordinal),
            cases = caseResults,
            wall_s = Math.Round(wall.Elapsed.TotalSeconds, 1),
        }, a.Get("out"));
        return ok ? 0 : 4;
    }
}

// =========================================================================================================
//  Recording observer + test policies
// =========================================================================================================

/// <summary>One decision as seen by the tests (public data only, plus the chosen key).</summary>
internal sealed class Rec
{
    public DecisionKind Kind;
    public bool Auto;
    public string? EventId;
    public string? Page;
    public bool Finished;
    public readonly List<string> Keys = new List<string>();
    public string ChosenKey = "";
    public int Gold;
    public int Hp;
    public int MaxHp;
    public bool PotionsUsable;
    public List<string>? Deck;
}

/// <summary>Coverage key of an action: what "this option" means across episodes.</summary>
internal static class Keys
{
    public static string Of(Decision d, in LegalAction a)
    {
        switch (d.Kind)
        {
            case DecisionKind.Event:
                if (a.Verb == Verb.Proceed)
                {
                    return "PROCEED(finished)";
                }
                if (a.Source.Kind == RefKind.EventOption)
                {
                    return a.Source.EventOption.TextKey;
                }
                return a.Verb.ToString();
            case DecisionKind.Shop:
                if (a.Source.Kind == RefKind.ShopEntry)
                {
                    return "BUY";
                }
                return a.Verb.ToString();
            case DecisionKind.Crystal:
                return "CLICK:" + (CrystalSphereMinigame.CrystalSphereToolType)a.Aux;
            case DecisionKind.Confirm:
                return a.Source.Index == 1 ? "YES" : "NO";
            case DecisionKind.Bundle:
                return a.Verb == Verb.Skip ? "SKIP" : "BUNDLE";
            default:
                return a.Verb.ToString();
        }
    }

    public static bool Covered(DecisionKind k) => k is DecisionKind.Event or DecisionKind.Shop or DecisionKind.Crystal or DecisionKind.Confirm or DecisionKind.Bundle;
}

internal sealed class Recorder : IEpisodeObserver
{
    public readonly List<Rec> Recs = new List<Rec>();
    public readonly List<string> Violations = new List<string>();
    public readonly SortedSet<string> LockedSeen = new SortedSet<string>(StringComparer.Ordinal);
    public EpisodeResult? Final;
    public bool Abandoned;
    public bool MapOpenAtEnd;
    public List<string> FinalDeck = new List<string>();
    public int FinalGold;
    public int FinalRelics;
    public Action<Recorder, RunContext, Decision, int, bool>? Extra;

    public void OnEpisodeStart(RunContext ctx)
    {
    }

    public void OnDecision(RunContext ctx, Decision d, int chosen, bool auto)
    {
        // the observation encoder's own consistency checks on every decision of every case and sweep episode
        // (test seam of the driver; plugged in by the executable for the area commands)
        string? audit = SpireHost.Driver.Tools.FrameAudit.Run(d);
        if (audit != null && Violations.Count < 20)
        {
            Violations.Add("frame audit: " + audit);
        }
        Player me = ctx.LocalPlayer;
        Rec r = new Rec
        {
            Kind = d.Kind,
            Auto = auto,
            Gold = me.Gold,
            Hp = me.Creature.CurrentHp,
            MaxHp = me.Creature.MaxHp,
            PotionsUsable = me.CanUseOrRemovePotions,
        };
        for (int i = 0; i < d.Actions.Count; i++)
        {
            r.Keys.Add(Keys.Of(d, d.Actions[i]));
        }
        r.ChosenKey = Keys.Of(d, d.Actions[chosen]);
        if (d.Context is EventContext ec)
        {
            r.EventId = ec.Event.Id.Entry;
            r.Page = ec.Event.Description?.LocEntryKey;
            r.Finished = ec.Finished;
            r.Deck = me.Deck.Cards.Select(c => c.Id.Entry).ToList();
            foreach (EventOptionInfo o in ec.Options)
            {
                if (o.IsLocked)
                {
                    LockedSeen.Add(o.TextKey);
                }
            }
            Invariants.CheckEvent(this, ctx, d, ec);
        }
        else if (d.Kind == DecisionKind.Crystal)
        {
            Invariants.CheckCrystal(this, ctx, d);
        }
        else if (d.Kind == DecisionKind.Shop && d.Context is ShopContext sc && sc.FakeMerchant)
        {
            Invariants.CheckFakeMerchant(this, ctx, d, sc);
        }
        Recs.Add(r);
        Extra?.Invoke(this, ctx, d, chosen, auto);
    }

    public void OnTerminalState(RunContext ctx, EpisodeResult provisional)
    {
        Abandoned = ctx.RunManager.IsAbandoned;
        MapOpenAtEnd = ctx.MapOpen;
        Player me = ctx.LocalPlayer;
        FinalDeck = me.Deck.Cards.Select(c => c.Id.Entry).ToList();
        FinalGold = me.Gold;
        FinalRelics = me.Relics.Count;
    }

    public void OnEpisodeEnd(EpisodeResult result)
    {
        Final = result;
    }

    public void Violation(string s)
    {
        if (Violations.Count < 20)
        {
            Violations.Add(s);
        }
    }
}

/// <summary>Per-decision invariants of the events team's frames (checked on every decision of every test episode).</summary>
internal static class Invariants
{
    private static readonly HashSet<string> PotionLockEvents = new HashSet<string>(StringComparer.Ordinal) { "RANWID_THE_ELDER", "STONE_OF_ALL_TIME", "THE_FUTURE_OF_POTIONS" };

    public static void CheckEvent(Recorder rec, RunContext ctx, Decision d, EventContext ec)
    {
        EventModel ev = ec.Event;
        EventContext view = ec; // contract R1: the event screen is in the context itself
        if (view.EventId != ev.Id.Entry || view.Finished != ev.IsFinished)
        {
            rec.Violation("EventContext out of date for " + ev.Id.Entry);
        }
        if (ev.IsFinished)
        {
            // The frame's OWN actions are exactly the synthesized PROCEED. Actions that a room-frame extension added
            // (LegalAction.Owner != null: the rooms module's out-of-combat potion use / discard, DECISIONS-H1 B.7) are
            // legitimate on a finished page and are not this frame's (ruling D1); only potion verbs may come from there.
            int own = 0;
            bool ownIsProceed = true;
            foreach (LegalAction a in d.Actions)
            {
                if (a.Owner == null)
                {
                    own++;
                    ownIsProceed &= a.Verb == Verb.Proceed;
                }
                else if (a.Verb != Verb.UsePotion && a.Verb != Verb.DiscardPotion)
                {
                    rec.Violation(ev.Id.Entry + ": a frame extension added " + a.Verb + " to a finished event frame");
                }
            }
            if (own != 1 || !ownIsProceed)
            {
                rec.Violation(ev.Id.Entry + ": finished event frame is not the single synthesized PROCEED");
            }
            if (view.Options.Count != 1 || !view.Options[0].Synthesized)
            {
                rec.Violation(ev.Id.Entry + ": finished event context does not carry exactly the synthesized PROCEED");
            }
        }
        else
        {
            int legal = 0;
            for (int i = 0; i < ev.CurrentOptions.Count; i++)
            {
                EventOption o = ev.CurrentOptions[i];
                bool shouldBeLegal = !o.IsLocked && !(o.WasChosen && o.DisableOnChosen);
                bool has = false;
                foreach (LegalAction a in d.Actions)
                {
                    if (a.Source.Kind == RefKind.EventOption && a.Source.Index == i)
                    {
                        has = true;
                    }
                }
                if (has != shouldBeLegal)
                {
                    rec.Violation(ev.Id.Entry + ": option " + o.TextKey + " legal=" + has + " expected " + shouldBeLegal);
                }
                legal += has ? 1 : 0;
                if (i < view.Options.Count && view.Options[i].Option != o)
                {
                    rec.Violation(ev.Id.Entry + ": EventContext option " + i + " mismatch");
                }
            }
            if (view.Options.Count != ev.CurrentOptions.Count)
            {
                rec.Violation(ev.Id.Entry + ": EventContext has " + view.Options.Count + " options, event " + ev.CurrentOptions.Count);
            }
        }
        if (PotionLockEvents.Contains(ev.Id.Entry))
        {
            // Locked from BeforeEventStarted until OnEventFinished (EnsureCleanup: SetEventFinished, or the room's Exit).
            // An event that runs out of options without SetEventFinished (TheFutureOfPotions with no potion, a state its
            // IsAllowed excludes) stays locked until the room is left - game behaviour, checked only while unfinished.
            if (!ev.IsFinished && ctx.LocalPlayer.CanUseOrRemovePotions)
            {
                rec.Violation(ev.Id.Entry + ": potions usable while the event is running");
            }
        }
    }

    public static void CheckCrystal(Recorder rec, RunContext ctx, Decision d)
    {
        if (d.Context is not CrystalContext cc)
        {
            rec.Violation("CRYSTAL decision without CrystalContext");
            return;
        }
        CrystalContext v = cc; // contract R2: the public view is in the context
        CrystalSphereMinigame m = cc.Minigame;
        int hidden = 0;
        foreach (CrystalSphereCell c in m.cells)
        {
            hidden += c.IsHidden ? 1 : 0;
        }
        if (d.Actions.Count != hidden * 2)
        {
            rec.Violation("CRYSTAL: " + d.Actions.Count + " actions for " + hidden + " hidden cells");
        }
        foreach (LegalAction a in d.Actions)
        {
            if (!((CrystalSphereCell)a.Source.Object!).IsHidden)
            {
                rec.Violation("CRYSTAL: action on a cleared cell");
            }
        }
        if (v.Divinations != m.DivinationCount || v.Cells.Count != m.cells.Length)
        {
            rec.Violation("CRYSTAL: view out of date");
            return;
        }
        foreach (CrystalCellView c in v.Cells)
        {
            if (c.Hidden && (c.ItemType != -2 || c.ItemIndex != -1))
            {
                rec.Violation("CRYSTAL: hidden cell carries item information");
            }
        }
        if (m.DivinationCount <= 0)
        {
            rec.Violation("CRYSTAL: frame asked with no divination left");
        }
    }

    public static void CheckFakeMerchant(Recorder rec, RunContext ctx, Decision d, ShopContext sc)
    {
        foreach (LegalAction a in d.Actions)
        {
            if (a.Source.Kind == RefKind.ShopEntry)
            {
                MegaCrit.Sts2.Core.Entities.Merchant.MerchantEntry e = a.Source.ShopEntry;
                if (e is not MegaCrit.Sts2.Core.Entities.Merchant.MerchantRelicEntry || !e.IsStocked || !e.EnoughGold)
                {
                    rec.Violation("FakeMerchant: illegal entry offered");
                }
            }
        }
        bool hasProceed = d.Actions.Any(a => a.Verb == Verb.Proceed);
        if (!hasProceed)
        {
            rec.Violation("FakeMerchant: no PROCEED");
        }
    }
}

/// <summary>SplitMix64, independent of every game RNG.</summary>
internal struct TestRng
{
    private ulong _s;

    public TestRng(ulong seed)
    {
        _s = seed * 0x9E3779B97F4A7C15UL + 0x632BE59BD9B4E019UL;
    }

    public int Next(int n)
    {
        ulong z = _s += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        z ^= z >> 31;
        return (int)(z % (ulong)n);
    }
}

/// <summary>
/// Coverage-directed test policy: on the events team's decision kinds it picks uniformly among the actions whose
/// coverage key (per event) has been chosen least often so far; elsewhere uniformly random. Optional preferences:
/// a list of keys to take in order whenever offered, and an "allow abandon" switch is the episode option's business.
/// </summary>
internal sealed class CoveragePolicy : IPolicy
{
    private TestRng _rng;
    private readonly Dictionary<string, int> _chosen;
    private readonly Queue<string>? _prefer;
    private readonly List<int> _best = new List<int>(64);

    public CoveragePolicy(ulong seed, Dictionary<string, int>? shared = null, IEnumerable<string>? prefer = null)
    {
        _rng = new TestRng(seed);
        _chosen = shared ?? new Dictionary<string, int>(StringComparer.Ordinal);
        _prefer = prefer != null ? new Queue<string>(prefer) : null;
    }

    public int Decide(Decision d)
    {
        if (_prefer != null && _prefer.Count > 0)
        {
            string want = _prefer.Peek();
            for (int i = 0; i < d.Actions.Count; i++)
            {
                if (Keys.Of(d, d.Actions[i]) == want)
                {
                    _prefer.Dequeue();
                    return Note(d, i);
                }
            }
        }
        if (!Keys.Covered(d.Kind))
        {
            return _rng.Next(d.Actions.Count);
        }
        string scope = Scope(d);
        int min = int.MaxValue;
        _best.Clear();
        for (int i = 0; i < d.Actions.Count; i++)
        {
            int n = _chosen.GetValueOrDefault(scope + Keys.Of(d, d.Actions[i]));
            if (n < min)
            {
                min = n;
                _best.Clear();
            }
            if (n == min)
            {
                _best.Add(i);
            }
        }
        return Note(d, _best[_rng.Next(_best.Count)]);
    }

    private int Note(Decision d, int i)
    {
        if (Keys.Covered(d.Kind))
        {
            string k = Scope(d) + Keys.Of(d, d.Actions[i]);
            _chosen[k] = _chosen.GetValueOrDefault(k) + 1;
        }
        return i;
    }

    private static string Scope(Decision d) => d.Context is EventContext ec ? ec.Event.Id.Entry + "|" : d.Kind + "|";
}

// =========================================================================================================
//  Part 1: targeted cases
// =========================================================================================================

internal sealed class Case
{
    public string Name = "";
    public int Seeds = 1;
    public string Character = "IRONCLAD";
    public Func<CaseState, RunContext, Task> Setup = (_, _) => Task.CompletedTask;
    public Func<ulong, IPolicy>? Policy;
    public string[]? Prefer;
    public bool AllowAbandon;
    public bool StopAfterSetup;

    /// <summary>Returns null when the case passed, else the failure reason.</summary>
    public Func<CaseState, Recorder, string?> Check = (_, _) => null;
}

internal sealed class CaseState
{
    public bool SetupDone;
    public readonly Dictionary<string, object> Values = new Dictionary<string, object>(StringComparer.Ordinal);
    public Recorder Rec = null!;
}

internal static class Cases
{
    public static (bool ok, object summary) Run(HostSession session, Case c, int seed, int hp)
    {
        CaseState st = new CaseState();
        Recorder rec = new Recorder();
        st.Rec = rec;
        string[] chars = { "IRONCLAD", "SILENT", "REGENT", "NECROBINDER", "DEFECT", "FOOL" };
        string character = c.Seeds > 1 ? chars[seed % chars.Length] : c.Character;
        IPolicy policy = c.Policy?.Invoke((ulong)(seed + 1)) ?? new CoveragePolicy((ulong)(seed + 1), prefer: c.Prefer);
        Func<RunContext, bool> stop = c.StopAfterSetup ? (ctx => st.SetupDone) : ScenarioTools.RoomFinished;
        long before = EventPatches.Counts.Values.Sum();
        EpisodeResult r = ScenarioRunner.Run(session, "case " + c.Name + " #" + seed, async ctx =>
        {
            await c.Setup(st, ctx);
            st.SetupDone = true;
        }, character, policy: policy, stopWhen: stop,
            options: new EpisodeOptions { DebugForcedHp = hp, AllowAbandon = c.AllowAbandon }, observer: rec);
        string? failure = null;
        if (r.Outcome == EpisodeOutcome.Aborted || r.TeardownFaults > 0)
        {
            failure = "aborted: " + (r.Abort?.Signature ?? "teardown fault") + " | " + r.Abort?.Message;
        }
        else if (rec.Violations.Count > 0)
        {
            failure = "invariant: " + rec.Violations[0];
        }
        else
        {
            failure = c.Check(st, rec);
        }
        bool ok = failure == null;
        return (ok, new
        {
            @case = c.Name,
            seed,
            character,
            passed = ok,
            failure,
            outcome = r.Outcome.ToString(),
            truncated_by = r.TruncatedBy,
            decisions = r.Counters.DecisionPoints,
            policy_decisions = r.Counters.PolicyDecisions,
            frames = string.Join(" ", rec.Recs.Select(x => Abbrev(x.Kind) + (x.Auto ? "*" : "") + ":" + x.ChosenKey).Take(40)),
            abort_frames = r.Abort?.Frames,
            diagnostics = r.Abort?.Diagnostics,
        });
    }

    private static string Abbrev(DecisionKind k) => k switch
    {
        DecisionKind.CardSelect => "CS",
        DecisionKind.CardReward => "CR",
        DecisionKind.Combat => "F",
        DecisionKind.Crystal => "X",
        DecisionKind.Confirm => "Y",
        _ => k.ToString().Substring(0, 1),
    };

    private static Task Act(RunContext ctx, int act) => ScenarioTools.EnterAct(ctx, act, doTransition: false);

    private static Rec? First(Recorder r, Func<Rec, bool> p) => r.Recs.FirstOrDefault(p);

    private static int IndexOf(Recorder r, Func<Rec, bool> p) => r.Recs.FindIndex(x => p(x));

    public static IEnumerable<Case> All()
    {
        // ---- bundles (CardSelectCmd.cs:657-698) -------------------------------------------------------------
        yield return new Case
        {
            Name = "bundle_scroll_boxes_pick_1",
            Seeds = 3,
            StopAfterSetup = true,
            Policy = seed => new PickPolicy(seed, DecisionKind.Bundle, 1),
            Setup = async (st, ctx) =>
            {
                uint before = ChoiceId(ctx);
                long hits = EventPatches.Counts.GetValueOrDefault("events-bundle-screen");
                List<CardModel> deckBefore = ctx.LocalPlayer.Deck.Cards.ToList();
                st.Rec.Extra = (rec, c, d, chosen, auto) =>
                {
                    if (d.Context is BundleContext bc)
                    {
                        st.Values["bundle"] = bc.Bundles[d.Actions[chosen].Source.Index];
                        st.Values["bundles"] = bc.Bundles.Count;
                        st.Values["canSkip"] = bc.CanSkip;
                    }
                };
                await ScenarioTools.GiveRelic(ctx, "SCROLL_BOXES");
                st.Values["choiceDelta"] = (int)(ChoiceId(ctx) - before);
                st.Values["screenHits"] = EventPatches.Counts.GetValueOrDefault("events-bundle-screen") - hits;
                st.Values["added"] = ctx.LocalPlayer.Deck.Cards.Where(c => !deckBefore.Contains(c)).ToList();
            },
            Check = (st, rec) =>
            {
                if (!st.Values.TryGetValue("bundle", out object? b))
                {
                    return "no BUNDLE decision";
                }
                if ((int)st.Values["bundles"] != 2 || (bool)st.Values["canSkip"])
                {
                    return "Scroll Boxes offers 2 bundles without skip";
                }
                IReadOnlyList<CardModel> bundle = (IReadOnlyList<CardModel>)b;
                List<CardModel> added = (List<CardModel>)st.Values["added"];
                if (added.Count != bundle.Count || bundle.Any(c => !added.Contains(c)))
                {
                    return "deck did not receive exactly the chosen bundle (" + added.Count + " added)";
                }
                if ((int)st.Values["choiceDelta"] != 1)
                {
                    return "ReserveChoiceId / SyncLocalChoice: choice id advanced by " + st.Values["choiceDelta"];
                }
                return (long)st.Values["screenHits"] == 1 ? null : "bundle screen stand-in not hit exactly once";
            },
        };
        yield return new Case
        {
            Name = "bundle_skip_path",
            StopAfterSetup = true,
            Policy = seed => new VerbPolicy(seed, DecisionKind.Bundle, Verb.Skip),
            Setup = async (st, ctx) =>
            {
                Player p = ctx.LocalPlayer;
                uint before = ChoiceId(ctx);
                List<IReadOnlyList<CardModel>> bundles = ScrollBoxes.GenerateRandomBundles(p, p.PlayerRng.Rewards)
                    .Select(b => (IReadOnlyList<CardModel>)b.Select(c => p.RunState.CreateCard(c, p)).ToList()).ToList();
                IEnumerable<CardModel> result = await CardSelectCmd.FromChooseABundleScreen(p, bundles, canSkip: true);
                st.Values["result"] = result.Count();
                st.Values["choiceDelta"] = (int)(ChoiceId(ctx) - before);
            },
            Check = (st, rec) =>
            {
                Rec? b = First(rec, x => x.Kind == DecisionKind.Bundle);
                if (b == null || !b.Keys.Contains("SKIP") || b.ChosenKey != "SKIP")
                {
                    return "no BUNDLE decision with SKIP taken";
                }
                if ((int)st.Values["result"] != 0)
                {
                    return "skip did not return an empty selection";
                }
                return (int)st.Values["choiceDelta"] == 1 ? null : "choice id advanced by " + st.Values["choiceDelta"];
            },
        };

        // ---- Crystal Sphere ------------------------------------------------------------------------------------
        foreach (string option in new[] { "CRYSTAL_SPHERE.pages.INITIAL.options.UNCOVER_FUTURE", "CRYSTAL_SPHERE.pages.INITIAL.options.PAYMENT_PLAN" })
        {
            int clicks = option.EndsWith("UNCOVER_FUTURE", StringComparison.Ordinal) ? 3 : 6;
            yield return new Case
            {
                Name = "crystal_sphere_" + (clicks == 3 ? "uncover_future" : "payment_plan"),
                Seeds = 4,
                Prefer = new[] { option },
                Setup = async (st, ctx) =>
                {
                    await Act(ctx, 1);
                    ScenarioTools.SetGold(ctx, 500);
                    st.Rec.Extra = (rec, c, d, chosen, auto) =>
                    {
                        if (d.Context is CrystalContext cc)
                        {
                            st.Values["minigame"] = cc.Minigame;
                        }
                    };
                    await ScenarioTools.JumpToEvent(ctx, "CRYSTAL_SPHERE");
                },
                Check = (st, rec) =>
                {
                    List<Rec> crystal = rec.Recs.Where(x => x.Kind == DecisionKind.Crystal).ToList();
                    if (crystal.Count != clicks)
                    {
                        return "expected " + clicks + " CRYSTAL decisions, got " + crystal.Count;
                    }
                    int lastCrystal = rec.Recs.FindLastIndex(x => x.Kind == DecisionKind.Crystal);
                    int reward = rec.Recs.FindIndex(lastCrystal + 1, x => x.Kind == DecisionKind.Reward);
                    int finished = rec.Recs.FindIndex(lastCrystal + 1, x => x.Kind == DecisionKind.Event && x.Finished);
                    // CompleteMinigame offers one reward per revealed item except the curse (CrystalSphereItem.ToReward;
                    // OneOffSynchronizer.OfferCrystalSphereRewards): an empty set raises no frame.
                    CrystalSphereMinigame m = (CrystalSphereMinigame)st.Values["minigame"];
                    int rewardable = m._revealed.Count(i => i is not MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereItems.CrystalSphereCurse);
                    if ((reward >= 0) != (rewardable > 0))
                    {
                        return "REWARD frame after the minigame: " + (reward >= 0) + ", rewardable revealed items: " + rewardable;
                    }
                    if (finished < 0 || finished < reward)
                    {
                        return "event did not finish after the rewards";
                    }
                    return rec.MapOpenAtEnd ? null : "map not opened";
                },
            };
        }

        // ---- Trial double-down ----------------------------------------------------------------------------------
        yield return new Case
        {
            Name = "trial_double_down_no",
            Seeds = 2,
            Prefer = new[] { "TRIAL.pages.INITIAL.options.REJECT", "TRIAL.pages.REJECT.options.DOUBLE_DOWN", "TRIAL.pages.REJECT.options.ACCEPT" },
            Setup = (st, ctx) => ScenarioTools.JumpToEvent(ctx, "TRIAL"),
            Check = (st, rec) =>
            {
                int confirm = IndexOf(rec, x => x.Kind == DecisionKind.Confirm);
                if (confirm < 0)
                {
                    return "no CONFIRM decision";
                }
                Rec c = rec.Recs[confirm];
                if (!c.Auto || c.Keys.Count != 1 || c.Keys[0] != "NO")
                {
                    return "CONFIRM should hold only NO (YES flagged Abandon) and be auto-advanced: " + string.Join(",", c.Keys);
                }
                Rec? next = rec.Recs.Skip(confirm + 1).FirstOrDefault(x => x.Kind == DecisionKind.Event);
                if (next == null || next.Page != "TRIAL.pages.REJECT.description" || !next.Keys.Contains("TRIAL.pages.REJECT.options.DOUBLE_DOWN"))
                {
                    return "after NO the REJECT page is not shown again";
                }
                return rec.Final?.Outcome == EpisodeOutcome.Loss ? "run lost" : (rec.MapOpenAtEnd ? null : "map not opened");
            },
        };
        yield return new Case
        {
            Name = "trial_double_down_yes_abandons",
            AllowAbandon = true,
            Policy = seed => new CoveragePolicy(seed, prefer: new[] { "TRIAL.pages.INITIAL.options.REJECT", "TRIAL.pages.REJECT.options.DOUBLE_DOWN", "YES" }),
            Setup = (st, ctx) => ScenarioTools.JumpToEvent(ctx, "TRIAL"),
            Check = (st, rec) =>
            {
                Rec? c = First(rec, x => x.Kind == DecisionKind.Confirm);
                if (c == null || c.Keys.Count != 2 || c.ChosenKey != "YES")
                {
                    return "CONFIRM with YES/NO and YES taken expected";
                }
                return rec.Final?.Outcome == EpisodeOutcome.Loss && rec.Abandoned ? null : "YES did not abandon the run (outcome " + rec.Final?.Outcome + ", abandoned " + rec.Abandoned + ")";
            },
        };

        // ---- FakeMerchant ---------------------------------------------------------------------------------------
        yield return new Case
        {
            Name = "fake_merchant_buy_and_leave",
            Seeds = 2,
            Prefer = new[] { "BUY", "BUY", "Proceed" },
            Setup = async (st, ctx) =>
            {
                await Act(ctx, 1);
                ScenarioTools.SetGold(ctx, 999);
                st.Values["relics"] = ctx.LocalPlayer.Relics.Count;
                await ScenarioTools.JumpToEvent(ctx, "FAKE_MERCHANT");
            },
            Check = (st, rec) =>
            {
                List<Rec> shop = rec.Recs.Where(x => x.Kind == DecisionKind.Shop).ToList();
                if (shop.Count != 3 || shop[0].Keys.Count(k => k == "BUY") != 6)
                {
                    return "expected 3 SHOP frames, first with 6 relic entries (got " + shop.Count + ")";
                }
                if (shop[1].Gold >= shop[0].Gold || shop[2].Gold >= shop[1].Gold)
                {
                    return "gold did not decrease with purchases";
                }
                if (rec.FinalRelics != (int)st.Values["relics"] + 2)
                {
                    return "relics not obtained";
                }
                return rec.MapOpenAtEnd ? null : "map not opened";
            },
        };
        yield return new Case
        {
            Name = "fake_merchant_foul_potion_fight",
            Seeds = 2,
            Prefer = new[] { "UsePotion" },
            Setup = async (st, ctx) =>
            {
                await Act(ctx, 1);
                ScenarioTools.SetGold(ctx, 0);
                await ScenarioTools.GivePotion(ctx, "FOUL_POTION");
                st.Rec.Extra = (rec, c, d, chosen, auto) =>
                {
                    if (d.Kind == DecisionKind.Combat && c.CurrentRoom is CombatRoom cr)
                    {
                        st.Values["encounter"] = cr.Encounter.Id.Entry;
                        st.Values["parent"] = cr.ParentEventId?.Entry ?? "-";
                    }
                    if (d.Context is RewardContext rc)
                    {
                        st.Values["rug"] = rc.Set.Rewards.Any(rw => rw is RelicReward rr && rr.Relic?.Id.Entry == "FAKE_MERCHANTS_RUG");
                    }
                };
                await ScenarioTools.JumpToEvent(ctx, "FAKE_MERCHANT");
            },
            Check = (st, rec) =>
            {
                Rec? shop = First(rec, x => x.Kind == DecisionKind.Shop);
                if (shop == null || !shop.Keys.Contains("UsePotion") || shop.Keys.Contains("BUY"))
                {
                    return "FakeMerchant SHOP frame without the FoulPotion throw (gold 0)";
                }
                if (!st.Values.TryGetValue("encounter", out object? enc) || (string)enc != "FAKE_MERCHANT_EVENT_ENCOUNTER" || (string)st.Values["parent"] != "FAKE_MERCHANT")
                {
                    return "FoulPotion did not start the FakeMerchant combat (" + (st.Values.GetValueOrDefault("encounter") ?? "none") + ")";
                }
                if (rec.Final?.Outcome == EpisodeOutcome.Loss)
                {
                    return null; // the fight was lost: also a legal ending
                }
                return st.Values.TryGetValue("rug", out object? rug) && (bool)rug ? null : "no reward set with FakeMerchantsRug";
            },
        };

        // ---- potion lock ----------------------------------------------------------------------------------------
        foreach ((string id, int act) in new[] { ("RANWID_THE_ELDER", 1), ("STONE_OF_ALL_TIME", 1), ("THE_FUTURE_OF_POTIONS", 0) })
        {
            yield return new Case
            {
                Name = "potion_lock_" + id.ToLowerInvariant(),
                Seeds = 2,
                Setup = async (st, ctx) =>
                {
                    await Act(ctx, act);
                    ScenarioTools.SetGold(ctx, 500);
                    foreach (PotionModel p in EventIds.TestPotions(2))
                    {
                        await ScenarioTools.GivePotion(ctx, p.Id.Entry);
                    }
                    foreach (RelicModel r in EventIds.TestRelics(2))
                    {
                        await ScenarioTools.GiveRelic(ctx, r.Id.Entry);
                    }
                    if (!ScenarioTools.Event(id).IsAllowed(ctx.Run))
                    {
                        throw new InvalidOperationException(id + " is not allowed in the constructed state");
                    }
                    await ScenarioTools.JumpToEvent(ctx, id);
                },
                Check = (st, rec) =>
                {
                    List<Rec> ev = rec.Recs.Where(x => x.EventId == id).ToList();
                    if (!ev.Any(x => !x.Finished && !x.PotionsUsable))
                    {
                        return "potions were not locked during the event";
                    }
                    return ev.Any(x => x.Finished && x.PotionsUsable) ? null : "potions not unlocked after the event";
                },
            };
        }

        // ---- Ancients ------------------------------------------------------------------------------------------
        // (The Ancient IsProceed "PROCEED" of AncientEventModel.cs:159-173 needs Hook.ShouldAllowAncient == false; no model
        //  overrides AbstractModel.ShouldAllowAncient in v0.113 and the WAX_CHOKER relic does not exist: unreachable, not tested.)
        foreach (AncientEventModel ancient in ModelDb.AllAncients.OrderBy(e => e.Id.Entry, StringComparer.Ordinal))
        {
            yield return new Case
            {
                Name = "ancient_" + ancient.Id.Entry.ToLowerInvariant(),
                Seeds = 3,
                Setup = (st, ctx) => ScenarioTools.JumpToEvent(ctx, ancient),
                Check = (st, rec) => First(rec, x => x.EventId == ancient.Id.Entry && x.Finished) != null ? null : ancient.Id.Entry + " did not finish",
            };
        }
        yield return new Case
        {
            Name = "neow_every_option",
            Seeds = 6,
            Setup = (st, ctx) => ScenarioTools.JumpToEvent(ctx, ModelDb.Event<Neow>()),
            Check = (st, rec) => First(rec, x => x.EventId == "NEOW" && x.Finished) != null ? null : "Neow did not finish",
        };
        yield return new Case
        {
            Name = "the_architect_wins",
            Seeds = 6,
            Setup = (st, ctx) => ScenarioTools.JumpToEvent(ctx, ModelDb.Event<TheArchitect>()),
            Check = (st, rec) => rec.Final?.Outcome == EpisodeOutcome.Win ? null : "TheArchitect did not end in Win (" + rec.Final?.Outcome + ")",
        };

        // ---- event combats: resume (BattlewornDummy) and no resume ------------------------------------------
        yield return new Case
        {
            Name = "event_combat_resume_battleworn_dummy",
            Seeds = 4,
            Setup = (st, ctx) => ScenarioTools.JumpToEvent(ctx, "BATTLEWORN_DUMMY"),
            Check = (st, rec) =>
            {
                int fight = IndexOf(rec, x => x.Kind == DecisionKind.Combat);
                if (fight < 0)
                {
                    return "no event combat";
                }
                if (rec.Final?.Outcome == EpisodeOutcome.Loss)
                {
                    return null;
                }
                Rec? back = rec.Recs.Skip(fight).FirstOrDefault(x => x.EventId == "BATTLEWORN_DUMMY");
                if (back == null || !back.Finished || (back.Page != "BATTLEWORN_DUMMY.pages.VICTORY.description" && back.Page != "BATTLEWORN_DUMMY.pages.DEFEAT.description"))
                {
                    return "the event was not resumed and finished after its combat (" + (back?.Page ?? "no event frame") + ")";
                }
                return rec.MapOpenAtEnd ? null : "map not opened";
            },
        };
        foreach ((string id, string[] path) in new[]
        {
            ("DENSE_VEGETATION", new[] { "DENSE_VEGETATION.pages.INITIAL.options.REST", "DENSE_VEGETATION.pages.REST.options.FIGHT" }),
            ("PUNCH_OFF", new[] { "PUNCH_OFF.pages.INITIAL.options.I_CAN_TAKE_THEM", "PUNCH_OFF.pages.I_CAN_TAKE_THEM.options.FIGHT" }),
            ("THE_LANTERN_KEY", Array.Empty<string>()),
        })
        {
            yield return new Case
            {
                Name = "event_combat_no_resume_" + id.ToLowerInvariant(),
                Seeds = 3,
                Prefer = path,
                Setup = (st, ctx) => ScenarioTools.JumpToEvent(ctx, id),
                Check = (st, rec) =>
                {
                    int fight = IndexOf(rec, x => x.Kind == DecisionKind.Combat);
                    if (fight < 0)
                    {
                        return path.Length == 0 ? null : "no event combat";
                    }
                    if (rec.Recs.Skip(fight).Any(x => x.EventId == id))
                    {
                        return "the event was shown again after a combat with shouldResumeAfterCombat=false";
                    }
                    return rec.Final?.Outcome == EpisodeOutcome.Loss || rec.MapOpenAtEnd ? null : "map not opened after the event combat";
                },
            };
        }

        // ---- contexts: what the encoder gets --------------------------------------------------------------------
        yield return new Case
        {
            Name = "context_waterfall_enter_page",
            Prefer = new[] { "WATERFALL_OF_OBVIOUS_SECRETS.pages.INITIAL.options.ENTER" },
            Setup = async (st, ctx) =>
            {
                st.Rec.Extra = (rec, c, d, chosen, auto) =>
                {
                    if (d.Context is EventContext ec && ec is { } v && v.PageKey == "WATERFALL_OF_OBVIOUS_SECRETS.pages.ENTER.description")
                    {
                        EventOptionInfo? nice = v.Options.FirstOrDefault(o => o.TextKey.EndsWith("BE_NICE", StringComparison.Ordinal));
                        EventOptionInfo? neg = v.Options.FirstOrDefault(o => o.TextKey.EndsWith("NEGOTIATE", StringComparison.Ordinal));
                        int cost = (int)ec.Event.DynamicVars["NegotiateCostKey"].BaseValue;
                        st.Values["nice_card"] = nice?.Cards.Any(cd => cd.Id.Entry == "DINGY_DAGGER") ?? false;
                        st.Values["neg_cost"] = neg?.Vars.Any(kv => kv.Value == cost) ?? false;
                        st.Values["legal"] = v.Options.Count(o => o.Legal);
                    }
                };
                await EventIds.Jump(ctx, "WATERFALL_OF_OBVIOUS_SECRETS");
            },
            Check = (st, rec) =>
            {
                if (!st.Values.ContainsKey("legal"))
                {
                    return "ENTER page not reached";
                }
                if (!(bool)st.Values["nice_card"])
                {
                    return "BE_NICE does not reference its hover-tip card DINGY_DAGGER";
                }
                if (!(bool)st.Values["neg_cost"])
                {
                    return "NEGOTIATE does not carry the printed NegotiateCostKey value";
                }
                return (int)st.Values["legal"] == 3 ? null : "expected 3 legal options";
            },
        };
        yield return new Case
        {
            Name = "context_neow_relic_options",
            Seeds = 3,
            Setup = async (st, ctx) =>
            {
                st.Rec.Extra = (rec, c, d, chosen, auto) =>
                {
                    if (d.Context is EventContext ec && ec is { } v && v.EventId == "NEOW" && !v.Finished)
                    {
                        st.Values["ancient"] = v.IsAncient;
                        st.Values["relics"] = v.Options.All(o => o.Relic != null);
                        st.Values["n"] = v.Options.Count;
                    }
                };
                await ScenarioTools.JumpToEvent(ctx, ModelDb.Event<Neow>());
            },
            Check = (st, rec) =>
            {
                if (!st.Values.ContainsKey("n"))
                {
                    return "no Neow frame";
                }
                if (!(bool)st.Values["ancient"] || !(bool)st.Values["relics"])
                {
                    return "Neow options must be flagged Ancient and carry their relic";
                }
                return (int)st.Values["n"] >= 2 ? null : "Neow offered fewer than 2 options";
            },
        };

        // ---- lethal options stay legal in single player --------------------------------------------------------
        yield return new Case
        {
            Name = "lethal_option_is_legal",
            Prefer = new[] { "DENSE_VEGETATION.pages.INITIAL.options.TRUDGE_ON" },
            Setup = async (st, ctx) =>
            {
                ScenarioTools.SetHp(ctx, 1);
                st.Rec.Extra = (rec, c, d, chosen, auto) =>
                {
                    if (d.Context is EventContext ec && ec is { } v && v.EventId == "DENSE_VEGETATION" && !v.Finished)
                    {
                        EventOptionInfo? o = v.Options.FirstOrDefault(x => x.TextKey.EndsWith("TRUDGE_ON", StringComparison.Ordinal));
                        st.Values["flag"] = o != null && o.WillKillPlayer && o.Legal;
                    }
                };
                await ScenarioTools.JumpToEvent(ctx, "DENSE_VEGETATION");
            },
            Check = (st, rec) =>
            {
                if (!st.Values.TryGetValue("flag", out object? f) || !(bool)f)
                {
                    return "TRUDGE_ON not flagged WillKillPlayer + legal";
                }
                return rec.Final?.Outcome == EpisodeOutcome.Loss ? null : "choosing the lethal option did not end the run";
            },
        };

        // ---- the seven fuzz-found event sites: the game logic after the presentation call still runs ----------
        yield return new Case
        {
            Name = "site_waterfall_enter",
            Seeds = 3,
            Prefer = new[] { "WATERFALL_OF_OBVIOUS_SECRETS.pages.INITIAL.options.ENTER" },
            Setup = (st, ctx) => EventIds.Jump(ctx, "WATERFALL_OF_OBVIOUS_SECRETS"),
            Check = (st, rec) =>
            {
                Rec? p = First(rec, x => x.Page == "WATERFALL_OF_OBVIOUS_SECRETS.pages.ENTER.description");
                return p != null && p.Keys.Count == 3 ? null : "ENTER page with 3 options (gold 500: NEGOTIATE unlocked) not reached";
            },
        };
        yield return new Case
        {
            Name = "site_dense_vegetation_rest",
            Seeds = 3,
            Prefer = new[] { "DENSE_VEGETATION.pages.INITIAL.options.REST" },
            Setup = (st, ctx) =>
            {
                ScenarioTools.SetHp(ctx, ctx.LocalPlayer.Creature.MaxHp / 2);
                return ScenarioTools.JumpToEvent(ctx, "DENSE_VEGETATION");
            },
            Check = (st, rec) =>
            {
                int i = IndexOf(rec, x => x.Page == "DENSE_VEGETATION.pages.REST.description");
                if (i < 0 || rec.Recs[i].Keys.Count != 1 || rec.Recs[i].Keys[0] != "DENSE_VEGETATION.pages.REST.options.FIGHT")
                {
                    return "REST page with FIGHT not reached";
                }
                Rec first = rec.Recs[IndexOf(rec, x => x.EventId == "DENSE_VEGETATION")];
                if (rec.Recs[i].Hp <= first.Hp)
                {
                    return "MimicRestSiteHeal did not heal";
                }
                return rec.Recs.Skip(i).Any(x => x.Kind == DecisionKind.Combat) ? null : "FIGHT did not start the event combat";
            },
        };
        foreach (string which in new[] { "STRIKES", "DEFENDS" })
        {
            yield return new Case
            {
                Name = "site_amalgamator_combine_" + which.ToLowerInvariant(),
                Seeds = 3,
                Prefer = new[] { "AMALGAMATOR.pages.INITIAL.options.COMBINE_" + which },
                Setup = (st, ctx) => ScenarioTools.JumpToEvent(ctx, "AMALGAMATOR"),
                Check = (st, rec) =>
                {
                    string card = which == "STRIKES" ? "ULTIMATE_STRIKE" : "ULTIMATE_DEFEND";
                    Rec? start = First(rec, x => x.EventId == "AMALGAMATOR");
                    Rec? done = First(rec, x => x.EventId == "AMALGAMATOR" && x.Finished);
                    if (start == null || done == null || !done.Deck!.Contains(card) || done.Deck.Count != start.Deck!.Count - 1)
                    {
                        return "two basics were not replaced by " + card;
                    }
                    return done.Page == "AMALGAMATOR.pages.COMBINE_" + which + ".description" ? null : "wrong final page " + done.Page;
                },
            };
        }
        yield return new Case
        {
            Name = "site_jungle_maze_safety_in_numbers",
            Seeds = 3,
            Prefer = new[] { "JUNGLE_MAZE_ADVENTURE.pages.INITIAL.options.JOIN_FORCES" },
            Setup = async (st, ctx) =>
            {
                st.Rec.Extra = (rec, c, d, chosen, auto) =>
                {
                    if (d.Context is EventContext ec && ec.Event is JungleMazeAdventure j && !ec.Finished)
                    {
                        st.Values["gold"] = (int)j.DynamicVars["JoinForcesGold"].BaseValue;
                    }
                };
                await EventIds.Jump(ctx, "JUNGLE_MAZE_ADVENTURE");
            },
            Check = (st, rec) =>
            {
                Rec? start = First(rec, x => x.EventId == "JUNGLE_MAZE_ADVENTURE");
                Rec? done = First(rec, x => x.EventId == "JUNGLE_MAZE_ADVENTURE" && x.Finished);
                if (start == null || done == null || done.Page != "JUNGLE_MAZE_ADVENTURE.pages.JOIN_FORCES.description")
                {
                    return "JOIN_FORCES page not reached";
                }
                return done.Gold - start.Gold == (int)st.Values["gold"] ? null : "gold gained " + (done.Gold - start.Gold) + " expected " + st.Values["gold"];
            },
        };
        yield return new Case
        {
            Name = "site_punch_off_nab",
            Seeds = 3,
            Prefer = new[] { "PUNCH_OFF.pages.INITIAL.options.NAB" },
            Setup = (st, ctx) => ScenarioTools.JumpToEvent(ctx, "PUNCH_OFF"),
            Check = (st, rec) =>
            {
                Rec? done = First(rec, x => x.EventId == "PUNCH_OFF" && x.Finished);
                if (done == null || !done.Deck!.Contains("INJURY") || done.Page != "PUNCH_OFF.pages.NAB.description")
                {
                    return "NAB: Injury / final page missing";
                }
                return rec.Recs.Any(x => x.Kind == DecisionKind.Reward) ? null : "NAB: relic reward not offered";
            },
        };
        yield return new Case
        {
            Name = "site_trial_accept",
            Seeds = 6,
            Prefer = new[] { "TRIAL.pages.INITIAL.options.ACCEPT" },
            Setup = (st, ctx) => ScenarioTools.JumpToEvent(ctx, "TRIAL"),
            Check = (st, rec) =>
            {
                Rec? p = First(rec, x => x.EventId == "TRIAL" && x.Keys.Any(k => k.EndsWith(".GUILTY", StringComparison.Ordinal)));
                if (p == null || p.Keys.Count != 2)
                {
                    return "ACCEPT did not reach a case page with GUILTY / INNOCENT";
                }
                return First(rec, x => x.EventId == "TRIAL" && x.Finished) != null ? null : "trial did not finish";
            },
        };
    }

    private static uint ChoiceId(RunContext ctx)
    {
        IReadOnlyList<uint> ids = ctx.RunManager.PlayerChoiceSynchronizer.ChoiceIds;
        int slot = ctx.SeatOf(ctx.LocalPlayer);
        return slot < ids.Count ? ids[slot] : 0;
    }
}

/// <summary>Takes action <c>index</c> (sorted order) on decisions of one kind; random elsewhere.</summary>
internal sealed class PickPolicy : IPolicy
{
    private TestRng _rng;
    private readonly DecisionKind _kind;
    private readonly int _index;

    public PickPolicy(ulong seed, DecisionKind kind, int index)
    {
        _rng = new TestRng(seed);
        _kind = kind;
        _index = index;
    }

    public int Decide(Decision d) => d.Kind == _kind && _index < d.Actions.Count ? _index : _rng.Next(d.Actions.Count);
}

/// <summary>Takes the first action with a given verb on decisions of one kind; random elsewhere.</summary>
internal sealed class VerbPolicy : IPolicy
{
    private TestRng _rng;
    private readonly DecisionKind _kind;
    private readonly Verb _verb;

    public VerbPolicy(ulong seed, DecisionKind kind, Verb verb)
    {
        _rng = new TestRng(seed);
        _kind = kind;
        _verb = verb;
    }

    public int Decide(Decision d)
    {
        if (d.Kind == _kind)
        {
            for (int i = 0; i < d.Actions.Count; i++)
            {
                if (d.Actions[i].Verb == _verb)
                {
                    return i;
                }
            }
        }
        return _rng.Next(d.Actions.Count);
    }
}

// =========================================================================================================
//  Part 2: the per-event sweep
// =========================================================================================================

internal static class Sweep
{
    internal sealed class Variant
    {
        public string Name = "";
        public int Gold;
        public double HpFraction = 1.0;  // of max HP; <= 0 -> 1 HP
        public int Potions;
        public bool FoulPotion;
        public int Relics;
        public int Floors;
        public bool ExhaustCard; // GraveOfTheForgotten: SoulsPower enchants Exhaust cards only
    }

    internal static readonly Variant[] Variants =
    {
        new Variant { Name = "base", Gold = 500, Potions = 2, Floors = 7 },
        new Variant { Name = "poor", Gold = 0 },
        new Variant { Name = "hurt", Gold = 200, HpFraction = 0.4, Potions = 1 },
        new Variant { Name = "lethal", Gold = 500, HpFraction = 0, Potions = 2 },
        new Variant { Name = "rich", Gold = 999, Potions = 2, FoulPotion = true, Relics = 5, Floors = 7, ExhaustCard = true },
    };

    internal sealed class EventStats
    {
        public string Id = "";
        public int Episodes;
        public int ExtraEpisodes;
        public int Passed;
        public int AllowedEpisodes;
        public int AllowedPassed;
        public readonly SortedSet<string> LegalSeen = new SortedSet<string>(StringComparer.Ordinal);
        public readonly SortedSet<string> Chosen = new SortedSet<string>(StringComparer.Ordinal);
        public readonly SortedSet<string> LockedSeen = new SortedSet<string>(StringComparer.Ordinal);
        public readonly SortedSet<string> AllowedIn = new SortedSet<string>(StringComparer.Ordinal);
        public readonly SortedSet<string> Kinds = new SortedSet<string>(StringComparer.Ordinal);
        public readonly List<string> Failures = new List<string>();
        public readonly Dictionary<string, int> Outcomes = new Dictionary<string, int>(StringComparer.Ordinal);
    }

    internal sealed class Result
    {
        public readonly List<EventStats> Events = new List<EventStats>();
        public int Episodes;
        public int Passed;
        public int AllowedEpisodes;
        public int AllowedPassed;
        public double WallS;
        public readonly SortedDictionary<string, int> FailuresByReason = new SortedDictionary<string, int>(StringComparer.Ordinal);

        public List<string> CoverageGaps()
        {
            List<string> gaps = new List<string>();
            foreach (EventStats e in Events)
            {
                foreach (string k in e.LegalSeen)
                {
                    if (!e.Chosen.Contains(k))
                    {
                        gaps.Add(e.Id + ":" + k);
                    }
                }
            }
            return gaps;
        }

        public List<string> NeverAllowed() => Events.Where(e => e.AllowedIn.Count == 0).Select(e => e.Id).ToList();

        public bool Ok => Passed == Episodes && CoverageGaps().Count == 0;

        public object Summary(bool brief) => new
        {
            events = Events.Count,
            episodes = Episodes,
            passed = Passed,
            allowed_episodes = AllowedEpisodes,
            allowed_passed = AllowedPassed,
            options_legal_seen = Events.Sum(e => e.LegalSeen.Count),
            options_chosen = Events.Sum(e => e.LegalSeen.Count(k => e.Chosen.Contains(k))),
            coverage_gaps = CoverageGaps(),
            never_allowed = NeverAllowed(),
            failures_by_reason = FailuresByReason,
            wall_s = Math.Round(WallS, 1),
            per_event = Events.Select(e => (object)new
            {
                id = e.Id,
                episodes = e.Episodes,
                extra_episodes = e.ExtraEpisodes,
                passed = e.Passed,
                allowed_in = e.AllowedIn,
                kinds = e.Kinds,
                outcomes = e.Outcomes,
                options = brief ? null : e.LegalSeen,
                locked_seen = brief ? null : e.LockedSeen,
                not_chosen = e.LegalSeen.Where(k => !e.Chosen.Contains(k)).ToList(),
                failures = e.Failures.Take(3).ToList(),
            }).ToList(),
        };
    }

    public static Result Run(HostSession session, Args a, int hp)
    {
        Stopwatch wall = Stopwatch.StartNew();
        Result result = new Result();
        int seeds = a.GetInt("seeds", 3);
        HashSet<string>? only = a.Get("events") is string ev && ev != "all" ? new HashSet<string>(ev.Split(','), StringComparer.OrdinalIgnoreCase) : null;
        HashSet<string>? variants = a.Get("variants") is string vs ? new HashSet<string>(vs.Split(','), StringComparer.OrdinalIgnoreCase) : null;
        string[] chars = a.Get("characters", "IRONCLAD,SILENT,REGENT,NECROBINDER,DEFECT,FOOL").Split(',');
        Dictionary<string, int> coverage = new Dictionary<string, int>(StringComparer.Ordinal);
        List<EventModel> events = EventIds.Every();
        int eventIndex = 0;
        foreach (EventModel canonical in events)
        {
            string id = canonical.Id.Entry;
            if (only != null && !only.Contains(id))
            {
                continue;
            }
            EventStats stats = new EventStats { Id = id };
            result.Events.Add(stats);
            foreach (Variant v in Variants)
            {
                if (variants != null && !variants.Contains(v.Name))
                {
                    continue;
                }
                for (int s = 0; s < seeds; s++)
                {
                    string character = chars[(eventIndex + s) % chars.Length];
                    RunOne(session, canonical, v, s, character, hp, coverage, stats, result);
                }
            }
            // Adaptive rounds: options drawn from large random pools (Neow, Pael ...) may have been legal once and lost
            // to another never-chosen option; keep sampling in the variants where the event is allowed until every
            // option that was ever legal has been chosen (bounded).
            int extra = 0;
            int maxExtra = a.GetInt("extra-rounds", 120);
            while (extra < maxExtra && stats.LegalSeen.Any(k => !stats.Chosen.Contains(k)))
            {
                Variant v = Variants[extra % Variants.Length];
                if (variants == null || variants.Contains(v.Name))
                {
                    int s = seeds + extra;
                    RunOne(session, canonical, v, s, chars[(eventIndex + s) % chars.Length], hp, coverage, stats, result);
                    stats.ExtraEpisodes++;
                }
                extra++;
            }
            eventIndex++;
        }
        result.WallS = wall.Elapsed.TotalSeconds;
        return result;
    }

    private static void RunOne(HostSession session, EventModel canonical, Variant v, int seed, string character, int hp,
        Dictionary<string, int> coverage, EventStats stats, Result result)
    {
        bool allowed = false;
        int act = 0;
        Recorder rec = new Recorder();
        string id = canonical.Id.Entry;
        EpisodeResult r = ScenarioRunner.Run(session, "sweep " + id + " " + v.Name + " #" + seed, async ctx =>
        {
            for (act = 0; act < ctx.Run.Acts.Count; act++)
            {
                if (act > 0)
                {
                    await ScenarioTools.EnterAct(ctx, act, doTransition: false);
                }
                await Apply(ctx, v);
                allowed = canonical.IsAllowed(ctx.Run);
                if (allowed || !ActGated(canonical, ctx))
                {
                    break;
                }
            }
            if (act >= ctx.Run.Acts.Count)
            {
                act = ctx.Run.Acts.Count - 1;
            }
            await ScenarioTools.JumpToEvent(ctx, canonical);
        }, character, policy: new CoveragePolicy((ulong)(seed * 7919 + 17), coverage), observer: rec,
            options: new EpisodeOptions { DebugForcedHp = hp, MaxPolicyDecisions = 20_000 });

        bool passed = ScenarioRunner.Passed(r) && rec.Violations.Count == 0;
        result.Episodes++;
        stats.Episodes++;
        if (allowed)
        {
            result.AllowedEpisodes++;
            stats.AllowedEpisodes++;
            stats.AllowedIn.Add(v.Name + "@act" + act);
        }
        if (passed)
        {
            result.Passed++;
            stats.Passed++;
            if (allowed)
            {
                result.AllowedPassed++;
                stats.AllowedPassed++;
            }
        }
        else
        {
            string reason = r.Abort?.Signature ?? (rec.Violations.Count > 0 ? "invariant: " + rec.Violations[0] : "truncated: " + r.TruncatedBy);
            result.FailuresByReason[reason] = result.FailuresByReason.GetValueOrDefault(reason) + 1;
            stats.Failures.Add(v.Name + "#" + seed + " " + character + ": " + reason + " | " + r.Abort?.Message + " | " + r.Abort?.Diagnostics);
        }
        string outcome = r.Outcome == EpisodeOutcome.Truncated ? r.TruncatedBy ?? "truncated" : r.Outcome.ToString();
        stats.Outcomes[outcome] = stats.Outcomes.GetValueOrDefault(outcome) + 1;
        foreach (Rec x in rec.Recs)
        {
            if (!Keys.Covered(x.Kind))
            {
                continue;
            }
            if (x.Kind == DecisionKind.Event && x.EventId != id)
            {
                continue; // e.g. an event a relic's AfterObtained opened (none in v0.113) - not this event
            }
            stats.Kinds.Add(x.Kind.ToString());
            foreach (string k in x.Keys)
            {
                stats.LegalSeen.Add(x.Kind + ":" + k);
            }
            stats.Chosen.Add(x.Kind + ":" + x.ChosenKey);
        }
        foreach (string k in rec.LockedSeen)
        {
            stats.LockedSeen.Add(k);
        }
    }

    /// <summary>True when raising the act could change EventModel.IsAllowed (the event reads CurrentActIndex).</summary>
    private static bool ActGated(EventModel canonical, RunContext ctx) => ActGatedIds.Contains(canonical.Id.Entry);

    private static readonly HashSet<string> ActGatedIds = new HashSet<string>(StringComparer.Ordinal)
    {
        // IsAllowed reads RunState.CurrentActIndex (Models.Events/*.cs IsAllowed)
        "CRYSTAL_SPHERE", "DOLL_ROOM", "FAKE_MERCHANT", "MUSEUM_OF_TORTURE", "POTION_COURIER", "RANWID_THE_ELDER",
        "RELIC_TRADER", "STONE_OF_ALL_TIME", "SYMBIOTE", "WELCOME_TO_WONGOS",
    };

    private static async Task Apply(RunContext ctx, Variant v)
    {
        Player p = ctx.LocalPlayer;
        ScenarioTools.SetGold(ctx, v.Gold);
        int target = v.HpFraction <= 0 ? 1 : Math.Max(1, (int)(p.Creature.MaxHp * v.HpFraction));
        ScenarioTools.SetHp(ctx, target);
        foreach (PotionModel potion in EventIds.TestPotions(v.Potions))
        {
            if (!p.Potions.Any(x => x.Id == potion.Id) && p.HasOpenPotionSlots)
            {
                await ScenarioTools.GivePotion(ctx, potion.Id.Entry);
            }
        }
        if (v.FoulPotion && !p.Potions.Any(x => x is MegaCrit.Sts2.Core.Models.Potions.FoulPotion) && p.HasOpenPotionSlots)
        {
            await ScenarioTools.GivePotion(ctx, "FOUL_POTION");
        }
        foreach (RelicModel relic in EventIds.TestRelics(v.Relics))
        {
            if (!p.Relics.Any(x => x.Id == relic.Id))
            {
                await ScenarioTools.GiveRelic(ctx, relic.Id.Entry);
            }
        }
        if (v.ExhaustCard && !p.Deck.Cards.Any(c => MegaCrit.Sts2.Core.Models.ModelDb.Enchantment<MegaCrit.Sts2.Core.Models.Enchantments.SoulsPower>().CanEnchant(c)))
        {
            CardModel? exhaust = MegaCrit.Sts2.Core.Models.ModelDb.AllCards
                .Where(c => c.Rarity == CardRarity.Common && c.Keywords.Contains(MegaCrit.Sts2.Core.Entities.Cards.CardKeyword.Exhaust))
                .OrderBy(c => c.Id.Entry, StringComparer.Ordinal).FirstOrDefault();
            if (exhaust != null)
            {
                await ScenarioTools.GiveCard(ctx, exhaust.Id.Entry);
            }
        }
        // TEST-ONLY state construction: PunchOff / SlipperyBridge read RunState.TotalFloor (= map-point history length).
        ScenarioTools.SetVisitedFloors(ctx, v.Floors);
        ScenarioTools.SetGold(ctx, v.Gold);
        ScenarioTools.SetHp(ctx, target);
    }
}
