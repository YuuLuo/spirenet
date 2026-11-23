using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereItems;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using SpireHost.Driver;
using SpireHost.Driver.Tools;
using SpireHost.Obs.Encoding;
using SpireHost.Obs.Schema;
using SpireHost.Obs.Vocab;

namespace SpireHost.Obs.Tests;

/// <summary>
/// TEST ONLY. SYNTHETIC CONFIRM / BUNDLE / CRYSTAL frames in controlled states (the real frames of the events module are
/// covered by obs --selftest "real_special" and by the leak scenarios): this sacrificial scenario builds the three
/// decisions exactly as the events module's frames do (CONFIRM = CHOOSE_OPTION(ConfirmChoice 0 / 1 + Abandon),
/// BUNDLE = CHOOSE_OPTION(GameRef.OfBundle(cards, i)) + SKIP, CRYSTAL = CRYSTAL_CLICK(cell y*w+x, aux = tool) for every
/// hidden cell x {Small 1, Big 2}, context = CrystalContext with its public view) over live game objects - a real
/// CrystalSphereMinigame (fresh Rng, no game stream used; some cells revealed by clearing IsHidden) - sealed with
/// Decision.SealForTest, and checks them:
///   * FrameCheck problems = 0, ACTIONS rows = actions, frames written to the golden file for the Python checker;
///   * CRYSTAL: revealed cells carry their item through the public view, hidden cells carry none;
///     P11 - removing / changing the items under HIDDEN cells leaves the bytes identical; positive control - revealing
///     one more cell changes them.
/// The scenario's game state is not reused afterwards (the episode is truncated).
/// </summary>
public static class SpecialFrames
{
    public static object Run(HostSession session, ObsRuntime.Loaded l, string? framesPath, out bool ok)
    {
        List<string> fails = new List<string>();
        Dictionary<string, object> facts = new Dictionary<string, object>();
        bool done = false;
        FileStream? fs = framesPath == null ? null : File.Create(framesPath);
        try
        {
            EpisodeResult r = ScenarioRunner.Run(session, "obs-special-frames", async ctx =>
            {
                ScenarioTools.SetHp(ctx, 9999, 9999);
                await ScenarioTools.JumpToRoom(ctx, RoomType.Monster);
            }, character: "IRONCLAD", policy: new RandomLegalPolicy(31), observer: new OnFirstDecision(ctx =>
            {
                if (done)
                {
                    return;
                }
                done = true;
                Build(ctx, l, fs, fails, facts);
            }), options: new EpisodeOptions { MaxPolicyDecisions = 3 });
            facts["episode"] = r.Outcome + "/" + r.TruncatedBy + (r.Abort == null ? "" : " " + r.Abort.Signature);
        }
        finally
        {
            fs?.Dispose();
        }
        if (!done)
        {
            fails.Add("scenario reached no decision");
        }
        ok = fails.Count == 0;
        return new { failures = fails, facts };
    }

    private sealed class OnFirstDecision : IEpisodeObserver
    {
        private readonly Action<RunContext> _a;

        public OnFirstDecision(Action<RunContext> a)
        {
            _a = a;
        }

        public void OnEpisodeStart(RunContext ctx)
        {
        }

        public void OnDecision(RunContext ctx, Decision decision, int chosen, bool auto) => _a(ctx);

        public void OnTerminalState(RunContext ctx, EpisodeResult provisional)
        {
        }

        public void OnEpisodeEnd(EpisodeResult result)
        {
        }
    }

    private static void Build(RunContext ctx, ObsRuntime.Loaded l, FileStream? fs, List<string> fails, Dictionary<string, object> facts)
    {
        ObsEncoder enc = ObsRuntime.NewEncoder(l);
        Player p = ctx.LocalPlayer;

        byte[] Encode(ObsEncoder e, Decision d, string label)
        {
            e.EncodeDecision(d, 1, 0, 0);
            byte[] b = e.Payload.ToArray();
            foreach (string pr in FrameCheck.Problems(b, out _))
            {
                fails.Add(label + ": " + pr);
            }
            if (ScriptTransport.Word(b, Hd.NActions) != d.Actions.Count)
            {
                fails.Add(label + ": ACTIONS rows != actions");
            }
            if (fs != null)
            {
                Span<byte> head = stackalloc byte[4];
                BinaryPrimitives.WriteInt32LittleEndian(head, b.Length);
                fs.Write(head);
                fs.Write(b);
            }
            return b;
        }

        // ---- CONFIRM (Trial.DoubleDown -> the abandon popup) ----
        Decision confirm = new Decision(DecisionKind.Confirm, null, 0, 0, new ConfirmContext { Key = ConfirmKeys.AbandonRun }) { Episode = ctx };
        confirm.Add(Verb.ChooseOption, 0, GameRef.OfConfirm(true), flags: ActionFlags.Abandon);
        confirm.Add(Verb.ChooseOption, 0, GameRef.OfConfirm(false));
        confirm.SealForTest(); // canonical order as the driver core produces it (obs request O6)
        byte[] cb = Encode(enc, confirm, "CONFIRM");
        facts["confirm_key_vocab"] = FrameCheckGlobal(cb, Gl.ConfirmKey);
        if (FrameCheckGlobal(cb, Gl.ConfirmKey) <= 1)
        {
            fails.Add("CONFIRM: confirm_key is PAD/UNK");
        }

        // ---- BUNDLE (CardSelectCmd.FromChooseABundleScreen) ----
        List<CardModel> deck = p.Deck.Cards.ToList();
        List<IReadOnlyList<CardModel>> bundles = new List<IReadOnlyList<CardModel>> { deck.Take(3).ToList(), deck.Skip(3).Take(3).ToList() };
        Decision bundle = new Decision(DecisionKind.Bundle, null, 0, 0, new BundleContext { Player = p, Bundles = bundles, CanSkip = true }) { Episode = ctx };
        for (int i = bundles.Count - 1; i >= 0; i--)
        {
            bundle.Add(Verb.ChooseOption, 0, GameRef.OfBundle(bundles[i], i));
        }
        bundle.Add(Verb.Skip, 0);
        bundle.SealForTest(); // -> SKIP (10) before CHOOSE_OPTION (11), bundle ordinals ascending
        byte[] bb = Encode(enc, bundle, "BUNDLE");
        facts["bundle_bytes"] = bb.Length;

        // ---- CRYSTAL ----
        CrystalSphereMinigame mg = new CrystalSphereMinigame(p, new Rng(77), 3);
        CrystalSphereCell[,] cells = mg.cells;
        int w = cells.GetLength(0), h = cells.GetLength(1);
        // reveal a band of cells (some will expose parts of items)
        for (int x = 2; x < 9; x++)
        {
            cells[x, 5].IsHidden = false;
        }
        Decision Crystal()
        {
            // the frame of the events module: the context carries the minigame (host-only) and the PUBLIC view rebuilt from it
            CrystalContext cc = new CrystalContext { Minigame = mg };
            cc.FillPublicView();
            Decision d = new Decision(DecisionKind.Crystal, null, 0, 0, cc) { Episode = ctx };
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    if (cells[x, y].IsHidden)
                    {
                        d.Add(Verb.CrystalClick, 0, GameRef.OfCrystalCell(cells[x, y], y * w + x), aux: (int)CrystalSphereMinigame.CrystalSphereToolType.Small);
                        d.Add(Verb.CrystalClick, 0, GameRef.OfCrystalCell(cells[x, y], y * w + x), aux: (int)CrystalSphereMinigame.CrystalSphereToolType.Big);
                    }
                }
            }
            return d.SealForTest();
        }
        byte[] v0 = Encode(enc, Crystal(), "CRYSTAL");
        FrameCheck.Problems(v0, out FrameCheck.View vv);
        int revealed = 0, unkRevealed = 0, itemCells = 0, hiddenWithItemData = 0;
        for (int r = 0; r < vv.Rows[T.Cells]; r++)
        {
            if (vv.Get(T.Cells, r, CellC.Hidden) == 0)
            {
                revealed++;
                if (vv.Get(T.Cells, r, CellC.ItemType) == VocabSet.UnkId)
                {
                    unkRevealed++;
                }
                if (vv.Get(T.Cells, r, CellC.ItemType) > 1)
                {
                    itemCells++;
                }
            }
            else if (vv.Get(T.Cells, r, CellC.ItemType) != -1 || vv.Get(T.Cells, r, CellC.ItemGood) != -1 || vv.Get(T.Cells, r, CellC.ItemIdx) != -1)
            {
                hiddenWithItemData++;
            }
        }
        facts["crystal_cells"] = vv.Rows[T.Cells];
        facts["crystal_actions"] = vv.Rows[T.Actions];
        facts["crystal_revealed"] = revealed;
        facts["crystal_revealed_item_cells"] = itemCells;
        if (revealed == 0 || unkRevealed != 0)
        {
            fails.Add($"CRYSTAL: {unkRevealed} of {revealed} revealed cells have item_type UNK (the public view names the item of every revealed cell)");
        }
        if (hiddenWithItemData != 0)
        {
            fails.Add($"CRYSTAL: {hiddenWithItemData} hidden cells carry item columns");
        }
        // P11: the items under hidden cells are hidden information
        List<(CrystalSphereCell cell, CrystalSphereItem? item)> saved = new List<(CrystalSphereCell, CrystalSphereItem?)>();
        CrystalSphereItem extra = new CrystalSphereCurse();
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                CrystalSphereCell c = cells[x, y];
                if (c.IsHidden)
                {
                    saved.Add((c, c.Item));
                    c.Item = c.Item == null ? extra : null;
                }
            }
        }
        byte[] v1 = Encode(enc, Crystal(), "CRYSTAL P11"); // the public view is rebuilt from the perturbed minigame
        foreach ((CrystalSphereCell cell, CrystalSphereItem? item) in saved)
        {
            cell.Item = item;
        }
        if (!LeakProbe.Same(v0, v1))
        {
            fails.Add("P11: items under hidden cells changed the CRYSTAL bytes");
        }
        facts["P11_hidden_cells_changed"] = saved.Count;
        // positive control: reveal one more cell
        CrystalSphereCell more = cells[5, 6];
        bool was = more.IsHidden;
        more.IsHidden = false;
        byte[] v2 = Encode(enc, Crystal(), "CRYSTAL control");
        more.IsHidden = was;
        if (LeakProbe.Same(v0, v2))
        {
            fails.Add("CRYSTAL control: revealing a cell did not change the bytes");
        }
    }

    private static int FrameCheckGlobal(byte[] f, int g) => ScriptTransport.GlobalOf(f, g);
}
