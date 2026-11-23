using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;

namespace SpireHost.Driver.Combat;

/// <summary>
/// The candidate list of a CardSelectCmd.From* prompt, computed from the captured arguments with the SAME statements
/// the game uses before it reaches a selector (Commands/CardSelectCmd.cs; the line of each case is cited below).
///
/// Why. For the local seat the game computes the list and hands it to ICardSelector.GetSelectedCards. For a REMOTE seat
/// the same From* call computes the same list and then throws it away: it only waits for the remote player's answer
/// (PlayerChoiceSynchronizer.WaitForRemoteChoice), because that player's own client computed the list for its own
/// screen. A virtual client has no second simulation, so the list of its prompt is rebuilt here, at the moment of the
/// wait: nothing runs between the game's own computation and that call, so the state is the one the game read.
///
/// How it is kept honest. Every prompt of the LOCAL seat is rebuilt as well and compared, element by element, with what
/// the game handed to the selector (HostCardSelector.GetSelectedCards): a difference aborts the episode
/// (HandlerFault/card_select_rebuild_mismatch). So every entry point that single-seat and seat-0 play reaches verifies
/// this code against the game on every prompt.
/// </summary>
public static class RemoteCandidates
{
    /// <summary>Calls / entries that are not rebuildable (must stay 0).</summary>
    public static long Rebuilds;
    public static long LocalChecks;
    public static long LocalMismatches;

    /// <summary>Null = the innermost entry of this prompt is not one of the entries that ask a selector.</summary>
    public static List<CardModel>? Rebuild(SelectionPrompt p)
    {
        Rebuilds++;
        Player player = p.Player;
        switch (p.InnerEntry)
        {
            case CardSelectEntry.ChooseACardScreen:
                // :64 / :75: selector.GetSelectedCards(cards, 0, 1)
                return p.GivenCards == null ? null : new List<CardModel>(p.GivenCards);
            case CardSelectEntry.SimpleGridForRewards:
                // :132 / :140: cards2.Select(c => c.Card)
                return p.GivenResults == null ? null : p.GivenResults.Select((CardCreationResult c) => c.Card).ToList();
            case CardSelectEntry.SimpleGrid:
                // :172: List<CardModel> cards = cardsIn.ToList()
                return p.GivenCards == null ? null : p.GivenCards.ToList();
            case CardSelectEntry.CombatPileFiltered:
            {
                // :250-260 candidates = filter == null ? pile.Cards : pile.Cards.Where(filter).ToList();
                // :273-278 / :286-291 a draw pile is shown "orderby x.Rarity, x" (its real order is hidden)
                CardPile? pile = p.Pile;
                if (pile == null)
                {
                    return null;
                }
                Func<CardModel, bool>? filter = p.Filter;
                IEnumerable<CardModel> enumerable = filter == null ? pile.Cards : pile.Cards.Where(filter).ToList();
                if (pile.Type == PileType.Draw)
                {
                    enumerable = from x in enumerable
                                 orderby x.Rarity, x
                                 select x;
                }
                return enumerable.ToList();
            }
            case CardSelectEntry.DeckForUpgrade:
                // :317
                return PileType.Deck.GetPile(player).Cards.Where((CardModel c) => c.IsUpgradable).ToList();
            case CardSelectEntry.DeckForTransformation:
                // :360
                return PileType.Deck.GetPile(player).Cards.Where((CardModel c) => c.Type != CardType.Quest && c.IsTransformable).ToList();
            case CardSelectEntry.DeckForEnchantmentCards:
            {
                // :425-431: the given cards in deck order
                IReadOnlyList<CardModel>? cards = p.GivenCards;
                if (cards == null)
                {
                    return null;
                }
                List<CardModel> list = new List<CardModel>();
                if (cards.Count > 0)
                {
                    Player owner = cards[0].Owner;
                    Dictionary<CardModel, int> indexMap = PileType.Deck.GetPile(owner).Cards.Select((CardModel card, int index) => new { card, index }).ToDictionary(x => x.card, x => x.index);
                    list = cards.OrderBy((CardModel c) => indexMap[c]).ToList();
                }
                return list;
            }
            case CardSelectEntry.DeckGeneric:
            {
                // :485-494
                List<CardModel> source = PileType.Deck.GetPile(player).Cards.ToList();
                Func<CardModel, bool>? filter = p.Filter;
                List<CardModel> list = filter == null ? source.ToList() : source.Where(filter).ToList();
                Func<CardModel, int>? sortingOrder = p.SortingOrder;
                if (sortingOrder != null)
                {
                    list = list.OrderBy(sortingOrder).ToList();
                }
                return list;
            }
            case CardSelectEntry.Hand:
            {
                // :554
                Func<CardModel, bool>? filter = p.Filter;
                return PileType.Hand.GetPile(player).Cards.Where(filter ?? ((CardModel _) => true)).ToList();
            }
            case CardSelectEntry.HandForUpgrade:
                // :622
                return PileType.Hand.GetPile(player).Cards.Where((CardModel c) => c.IsUpgradable).ToList();
            default:
                return null;
        }
    }

    /// <summary>The first position at which the two lists differ; -1 = identical (same instances, same order).</summary>
    public static int FirstDifference(IReadOnlyList<CardModel> rebuilt, IReadOnlyList<CardModel> game)
    {
        int n = Math.Min(rebuilt.Count, game.Count);
        for (int i = 0; i < n; i++)
        {
            if (!ReferenceEquals(rebuilt[i], game[i]))
            {
                return i;
            }
        }
        return rebuilt.Count == game.Count ? -1 : n;
    }

    /// <summary>
    /// The answer of a prompt as the LOCAL branch of its entry point encodes it for
    /// PlayerChoiceSynchronizer.SyncLocalChoice, i.e. what the remote branch of the same entry point decodes:
    ///   ChooseACardScreen     FromIndex(cards.IndexOf(result))       (:89-91; no card = -1)        -> AsIndex (:96-97)
    ///   SimpleGrid(ForRewards) FromIndexes(index of each pick in the list handed over) (:148-149, :206-207) -> AsIndexes (:154, :212)
    ///   CombatPile, Hand      FromMutableCombatCards(result)         (:300, :578)                  -> AsCombatCards (:305, :583)
    ///   HandForUpgrade        FromMutableCombatCard(result)          (:642)                        -> AsCombatCards().FirstOrDefault() (:647)
    ///   Deck*                 FromMutableDeckCards(result)           (:346, :393, :461, :524)      -> AsDeckCards (:351, :398, :466, :529)
    /// <paramref name="raw"/> = the candidate list in the order the game's entry point holds it (indexes refer to it);
    /// <paramref name="picked"/> = the chosen instances in pick order, empty for skip / cancel.
    /// </summary>
    public static PlayerChoiceResult Encode(SelectionPrompt p, IReadOnlyList<CardModel> raw, IReadOnlyList<CardModel> picked)
    {
        switch (p.InnerEntry)
        {
            case CardSelectEntry.ChooseACardScreen:
                return PlayerChoiceResult.FromIndex(picked.Count == 0 ? -1 : IndexIn(raw, picked[0]));
            case CardSelectEntry.SimpleGridForRewards:
            case CardSelectEntry.SimpleGrid:
            {
                List<int> indexes = new List<int>(picked.Count);
                for (int i = 0; i < picked.Count; i++)
                {
                    indexes.Add(IndexIn(raw, picked[i]));
                }
                return PlayerChoiceResult.FromIndexes(indexes);
            }
            case CardSelectEntry.CombatPileFiltered:
            case CardSelectEntry.Hand:
                return PlayerChoiceResult.FromMutableCombatCards(picked);
            case CardSelectEntry.HandForUpgrade:
                return PlayerChoiceResult.FromMutableCombatCard(picked.Count == 0 ? null : picked[0]);
            case CardSelectEntry.DeckForUpgrade:
            case CardSelectEntry.DeckForTransformation:
            case CardSelectEntry.DeckForEnchantmentCards:
            case CardSelectEntry.DeckGeneric:
                return PlayerChoiceResult.FromMutableDeckCards(picked);
            default:
                throw new InvalidOperationException("No remote answer encoding for " + p.InnerEntry + " (" + p + ").");
        }
    }

    /// <summary>The cards the REMOTE branch of the entry point makes of a received answer (the As* column of <see cref="Encode"/>).</summary>
    public static List<CardModel> Decode(SelectionPrompt p, IReadOnlyList<CardModel> raw, PlayerChoiceResult answer)
    {
        List<CardModel> cards = new List<CardModel>(4);
        switch (p.InnerEntry)
        {
            case CardSelectEntry.ChooseACardScreen:
            {
                int num = answer.AsIndex();
                if (num >= 0)
                {
                    cards.Add(raw[num]);
                }
                break;
            }
            case CardSelectEntry.SimpleGridForRewards:
            case CardSelectEntry.SimpleGrid:
                foreach (int i in answer.AsIndexes())
                {
                    cards.Add(raw[i]);
                }
                break;
            case CardSelectEntry.CombatPileFiltered:
            case CardSelectEntry.Hand:
                cards.AddRange(answer.AsCombatCards());
                break;
            case CardSelectEntry.HandForUpgrade:
            {
                CardModel? first = answer.AsCombatCards().FirstOrDefault();
                if (first != null)
                {
                    cards.Add(first);
                }
                break;
            }
            case CardSelectEntry.DeckForUpgrade:
            case CardSelectEntry.DeckForTransformation:
            case CardSelectEntry.DeckForEnchantmentCards:
            case CardSelectEntry.DeckGeneric:
                cards.AddRange(answer.AsDeckCards());
                break;
            default:
                throw new InvalidOperationException("No remote answer decoding for " + p.InnerEntry + " (" + p + ").");
        }
        return cards;
    }

    public static long EncodingChecks;
    public static long EncodingMismatches;

    private static readonly PacketWriter _writer = new PacketWriter { WarnOnGrow = false };
    private static readonly PacketReader _reader = new PacketReader();

    /// <summary>
    /// The answer of a LOCAL prompt, sent the way a remote seat's answer travels: encoded (<see cref="Encode"/>), written
    /// and read with the game's packet serializer, turned back into a PlayerChoiceResult (PlayerChoiceResult.FromNetData,
    /// what PlayerChoiceSynchronizer.WaitForRemoteChoice returns, :107-110) and decoded (<see cref="Decode"/>). The cards
    /// that come out must be the cards that were picked, in pick order. Null = they are; else what differs. Read-only.
    /// </summary>
    public static string? CheckEncoding(RunContext ctx, SelectionPrompt p, IReadOnlyList<CardModel> raw, IReadOnlyList<CardModel> picked)
    {
        EncodingChecks++;
        NetPlayerChoiceResult net = Encode(p, raw, picked).ToNetData();
        _writer.Reset();
        _writer.Write(net);
        _reader.Reset(_writer.Buffer);
        NetPlayerChoiceResult received = _reader.Read<NetPlayerChoiceResult>();
        List<CardModel> back = Decode(p, raw, PlayerChoiceResult.FromNetData(p.Player, ctx.Run, received));
        int differs = FirstDifference(back, picked);
        if (differs < 0)
        {
            return null;
        }
        EncodingMismatches++;
        return $"the answer ({picked.Count} cards) comes back as {back.Count} cards from its wire form, first difference at pick {differs}";
    }

    private static int IndexIn(IReadOnlyList<CardModel> raw, CardModel card)
    {
        for (int i = 0; i < raw.Count; i++)
        {
            if (ReferenceEquals(raw[i], card))
            {
                return i;
            }
        }
        throw new InvalidOperationException("A picked card is not a candidate of its prompt.");
    }
}

/// <summary>
/// The router of CardSelectCmd prompts for remote seats (multi-seat only): the wait belongs to the From* call the current
/// async flow is inside of (<see cref="SelectionCapture.Current"/>, the same association the local selector uses) when
/// that prompt is the waiting player's and has not been asked yet.
/// </summary>
public sealed class CardSelectRemoteRouter : IRemoteChoiceRouter
{
    private readonly CombatLifecycle _lifecycle;

    public CardSelectRemoteRouter(CombatLifecycle lifecycle)
    {
        _lifecycle = lifecycle;
    }

    public string Name => "combat:card-select-remote";

    public bool TryRoute(RunContext ctx, Player player, uint choiceId)
    {
        SelectionPrompt? prompt = SelectionCapture.Current;
        if (prompt == null || prompt.Asked || !ReferenceEquals(prompt.Player, player))
        {
            return false;
        }
        int seat = ctx.SeatOf(player);
        HostCardSelector? slot = seat >= 0 && seat < _lifecycle.Selectors.Length ? _lifecycle.Selectors[seat] : null;
        if (slot == null || !ctx.IsRemoteSeat(seat) || ctx.Mp == null)
        {
            return false;
        }
        slot.OpenRemote(prompt, ctx.Mp.Client(seat), choiceId);
        return true;
    }
}
